"""Debug script for testing approach phase motion."""
import numpy as np
import os
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from src.config import init_genesis, parse_args
from src.sensors import SensorDataLogger
from src.scene import setup_scene, setup_entities, setup_camera, configure_robot
from src.camera import update_wrist_camera
from src.utils import print_joint_state, print_ik_target_vs_current, check_joint_limits_violated


def debug_approach_phase():
    """Test the approach phase in isolation for debugging flailing."""
    init_genesis()
    
    # Setup
    logger = SensorDataLogger()
    scene = setup_scene(show_viewer=True)
    
    # Single cube at fixed position
    cube_pos = np.array([0.75, 0.0, 0.02])
    plane, cubes, franka = setup_entities(scene, num_cubes=1, cube_positions=[cube_pos])
    cam = setup_camera(scene)
    
    scene.build()
    
    # Configure robot
    motors_dof = np.arange(7)
    fingers_dof = np.arange(7, 9)
    configure_robot(franka)
    
    time.sleep(0.5)
    
    end_effector = franka.get_link('hand')
    if cam is not None:
        update_wrist_camera(cam, end_effector)
    
    # Test parameters
    quat_ny = np.array([0, 1, 0, 0])  # face-down
    hover_height = 0.25
    approach_height = 0.12
    
    # === PHASE 1: Move to hover ===
    print("\n" + "="*70)
    print("PHASE 1: HOVER")
    print("="*70)
    
    hover_target_pos = np.array([cube_pos[0], cube_pos[1], hover_height])
    print(f"Target: {hover_target_pos}")
    
    q_hover = franka.inverse_kinematics(
        link=end_effector,
        pos=hover_target_pos,
        quat=quat_ny,
    )
    q_hover[-2:] = 0.04
    print(f"IK result: {q_hover}")
    print_ik_target_vs_current(franka, q_hover, None, "Hover")
    
    # Move to hover with planning
    path = franka.plan_path(qpos_goal=q_hover, num_waypoints=200)
    print(f"Planned trajectory: {len(path)} waypoints")
    
    for step_idx, waypoint in enumerate(path):
        franka.control_dofs_position(waypoint)
        scene.step()
        update_wrist_camera(cam, end_effector)
        
        if step_idx % 50 == 0:
            print(f"  Step {step_idx}/{len(path)}")
        
        if step_idx == len(path) - 1:
            print_joint_state(franka, "Hover (Final)", step_idx)
    
    # Stabilize at hover
    print("\nStabilizing at hover for 60 steps...")
    for i in range(60):
        scene.step()
        update_wrist_camera(cam, end_effector)
        if i == 59:
            print_joint_state(franka, "Hover (Stabilized)", i)
    
    # === PHASE 2: Approach ===
    print("\n" + "="*70)
    print("PHASE 2: APPROACH")
    print("="*70)
    
    lateral_offset = np.array([0.0, 0.0])  # No offset for debug
    height_jitter = 0.0  # No jitter for debug
    
    approach_target_pos = np.array([
        cube_pos[0] + lateral_offset[0],
        cube_pos[1] + lateral_offset[1],
        approach_height + height_jitter,
    ])
    
    print(f"Target: {approach_target_pos}")
    print(f"Lateral offset: {lateral_offset}")
    print(f"Height jitter: {height_jitter}")
    
    q_approach = franka.inverse_kinematics(
        link=end_effector,
        pos=approach_target_pos,
        quat=quat_ny,
    )
    
    print(f"IK result: {q_approach}")
    q_current = franka.get_qpos()
    print_ik_target_vs_current(franka, q_approach[:-2], q_current, "Approach")
    
    # Execute approach with detailed debug
    print(f"\nExecuting approach for 90 steps...")
    print(f"Target joint positions (arm): {q_approach[:-2]}")
    
    for i in range(90):
        franka.control_dofs_position(q_approach)
        scene.step()
        update_wrist_camera(cam, end_effector)
        
        if i % 30 == 0:
            print_joint_state(franka, f"Approach", i)
        
        # Check for joint limit violations
        arm_lower = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
        arm_upper = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])
        if check_joint_limits_violated(franka, arm_lower, arm_upper):
            print(f"[FLAILING DETECTED] Joint limits violated at step {i}")
            break
    
    print_joint_state(franka, "Approach (Final)", 89)
    
    print("\n" + "="*70)
    print("Debug complete. Check output above for joint state and velocity issues.")
    print("="*70)


if __name__ == "__main__":
    debug_approach_phase()
