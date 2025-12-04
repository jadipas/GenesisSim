import genesis as gs
import numpy as np
import time

# --- 1. CONFIGURATION ---
BACKEND = gs.cuda

# --- 2. INITIALIZATION ---
print(f"Initializing Genesis with backend: {BACKEND.name}")
gs.init(backend=BACKEND)

# --- 3. SCENE SETUP ---
scene = gs.Scene(show_viewer=True)

# Add a ground plane
plane = scene.add_entity(gs.morphs.Plane()) 

# Load the Franka Emika Panda with IK enabled
franka = scene.add_entity(
    gs.morphs.MJCF(
        file='xml/franka_emika_panda/panda.xml',
        pos=np.array([0.0, 0.0, 0.05]),
        requires_jac_and_IK=True  # Enable Jacobian and IK
    )
)

# Finalize the scene compilation
scene.build()

# =========================================================================
# SETUP IK-BASED CONTROL FOR PICK-AND-PLACE
# =========================================================================

# 1. Set PD Control Gains for smooth arm motion
kp_array = np.array([4500.0] * 7 + [2500.0] * 2) 
kv_array = np.array([450.0] * 7 + [250.0] * 2)
franka.set_dofs_kp(kp_array)
franka.set_dofs_kv(kv_array)

# Set force limits
franka.set_dofs_force_range(
    lower=np.array([-87, -87, -87, -87, -12, -12, -12, -100, -100]), 
    upper=np.array([ 87,  87,  87,  87,  12,  12,  12,  100,  100])
)

# 2. Get the end-effector link (tool0 in Franka, but we can use link7 as reference)
# For the Panda, "link7" is the wrist, and we attach an imaginary EE at a fixed offset
ee_link = franka.get_link("link7")

# 3. Define pick and place waypoints (in world coordinates)
# Adjust these based on your scene layout
PICK_APPROACH = np.array([0.3, 0.1, 0.15])    # Approach pose (above the object)
PICK_TARGET = np.array([0.3, 0.1, 0.05])     # Pick pose (grasp location)
PLACE_APPROACH = np.array([0.5, 0.1, 0.15])  # Intermediate pose
PLACE_TARGET = np.array([0.5, 0.1, 0.05])    # Place pose
HOME = np.array([0.0, 0.0, 0.4])             # Safe home position

# 4. Target orientation (gripper pointing downward)
# Quaternion: [w, x, y, z] pointing downward (Z-axis points down)
GRASP_QUAT = np.array([0.7071, 0.7071, 0.0, 0.0])  # 90° rotation around X-axis

# 5. Task state machine
task_phase = 0
phase_timer = 0.0
phase_duration = 2.0  # seconds per phase
current_qpos = None
gripper_state = 0.04  # Start open

print(f"Scene built successfully. Franka EE link: {ee_link.name}")
print(f"Pick target: {PICK_TARGET}, Place target: {PLACE_TARGET}")
print("Starting pick-and-place simulation...")

# --- 4. SIMULATION LOOP ---
SIM_DURATION = 30.0 
STEPS_PER_SECOND = 1000 
NUM_STEPS = int(SIM_DURATION * STEPS_PER_SECOND)

start_time = time.time()

try:
    for i in range(NUM_STEPS):
        t = i / STEPS_PER_SECOND
        phase_timer += 1.0 / STEPS_PER_SECOND
        
        # Determine current task phase and target EE pose
        if phase_timer >= phase_duration:
            phase_timer = 0.0
            task_phase = (task_phase + 1) % 6
        
        # Define waypoints for the pick-and-place task
        if task_phase == 0:
            # Phase 0: Move to pick approach
            target_pos = PICK_APPROACH
            target_quat = GRASP_QUAT
            gripper_state = 0.04  # Keep open
            phase_name = "Pick Approach"
        elif task_phase == 1:
            # Phase 1: Move down to pick target
            target_pos = PICK_TARGET
            target_quat = GRASP_QUAT
            gripper_state = 0.04  # Still open
            phase_name = "Pick Grasp"
        elif task_phase == 2:
            # Phase 2: Close gripper
            target_pos = PICK_TARGET
            target_quat = GRASP_QUAT
            gripper_state = 0.0  # Close gripper
            phase_name = "Gripper Close"
        elif task_phase == 3:
            # Phase 3: Lift and move to place approach
            target_pos = PLACE_APPROACH
            target_quat = GRASP_QUAT
            gripper_state = 0.0  # Keep closed
            phase_name = "Place Approach"
        elif task_phase == 4:
            # Phase 4: Move down to place target
            target_pos = PLACE_TARGET
            target_quat = GRASP_QUAT
            gripper_state = 0.0  # Keep closed
            phase_name = "Place Position"
        else:  # task_phase == 5
            # Phase 5: Open gripper and return home
            if phase_timer < phase_duration * 0.5:
                target_pos = PLACE_TARGET
                target_quat = GRASP_QUAT
                gripper_state = 0.04  # Open gripper
                phase_name = "Gripper Open"
            else:
                target_pos = HOME
                target_quat = GRASP_QUAT
                gripper_state = 0.04
                phase_name = "Return Home"
        
        # Call IK solver to get joint angles for the target EE pose
        try:
            qpos_arm = franka.inverse_kinematics(
                link=ee_link,
                pos=target_pos,
                quat=target_quat,
                init_qpos=current_qpos,
                respect_joint_limit=True,
                max_samples=20,
                max_solver_iters=20,
                damping=0.01,
                pos_tol=1e-3,
                rot_tol=1e-2,
            )
            # qpos_arm is 7-DoF; append gripper commands
            qpos_full = np.concatenate([qpos_arm, [gripper_state, gripper_state]])
            current_qpos = qpos_arm  # Use this as next init for smoother IK
        except:
            # If IK fails, hold current position
            if current_qpos is None:
                qpos_full = np.array([0, 0, 0, -1.0, 0, 0, 0, 0.04, 0.04])
            else:
                qpos_full = np.concatenate([current_qpos, [gripper_state, gripper_state]])
        
        # Send control command
        franka.control_dofs_position(qpos_full)
        
        # Print progress every 1 second
        if i % 1000 == 0:
            ee_pos = ee_link.get_pos().cpu().numpy()
            print(f"t={t:5.1f}s | Phase: {phase_name:15s} | EE: [{ee_pos[0]:.3f}, {ee_pos[1]:.3f}, {ee_pos[2]:.3f}] → Target: [{target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f}]")
        
        scene.step()
        
except KeyboardInterrupt:
    print("Simulation interrupted by user.")
except Exception as e:
    print(f"An error occurred during simulation: {e}")

end_time = time.time()
print(f"Simulation finished after {NUM_STEPS} steps in {end_time - start_time:.2f} seconds.")