import numpy as np
import genesis as gs
import time
import cv2 
import argparse
from collections import defaultdict
from typing import Dict, List, Any

########################## command line arguments ##########################
def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Genesis Franka Pick-and-Place Simulation')
    parser.add_argument('--headless', action='store_true', 
                       help='Run without viewer and without camera playback')
    parser.add_argument('--sim-only', action='store_true',
                       help='Run with viewer but without camera playback')
    return parser.parse_args()

args = parse_args()

########################## init ##########################
gs.init(backend=gs.gpu)

########################## sensor data logger ##########################
class SensorDataLogger:
    """
    Collects and stores sensor data for slip detection and dataset creation.
    
    Records:
    - Proprioception: joint states, EE pose/twist, gripper state
    - Contact info: ground-truth contact for auto-labeling
    - Vision: RGBD images (optional)
    """
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Clear all logged data."""
        self.data = defaultdict(list)
        self.timestep = 0
    
    def log_step(self, step_data: Dict[str, Any]):
        """Log data for current timestep."""
        step_data['timestep'] = self.timestep
        for key, value in step_data.items():
            self.data[key].append(value)
        self.timestep += 1
    
    def get_last(self, key: str) -> Any:
        """Get last logged value for a key."""
        if key in self.data and len(self.data[key]) > 0:
            return self.data[key][-1]
        return None
    
    def save(self, filepath: str):
        """Save logged data to npz file."""
        save_dict = {}
        for key, values in self.data.items():
            if isinstance(values[0], np.ndarray):
                save_dict[key] = np.array(values)
            else:
                save_dict[key] = values
        np.savez_compressed(filepath, **save_dict)
        print(f"Saved sensor data to {filepath}")
    
    def get_contact_state_label(self) -> int:
        """
        Get contact state label for current timestep.
        Returns: 0 = no contact, 1 = in contact
        """
        return self.get_last('in_contact') or 0
    
    def detect_contact_lost(self) -> bool:
        """
        Detect if contact was just lost (for event labeling).
        Returns: True if contact lost this timestep
        """
        if len(self.data['in_contact']) < 2:
            return False
        prev_contact = self.data['in_contact'][-2]
        curr_contact = self.data['in_contact'][-1]
        return prev_contact and not curr_contact
    
    def get_summary_stats(self) -> Dict[str, int]:
        """Get summary statistics of collected data."""
        total_timesteps = self.timestep
        total_contact = sum(self.data['in_contact'])
        contact_lost_events = sum([
            self.data['in_contact'][i-1] and not self.data['in_contact'][i] 
            for i in range(1, len(self.data['in_contact']))
        ])
        return {
            'total_timesteps': total_timesteps,
            'total_contact': total_contact,
            'contact_lost_events': contact_lost_events
        }

########################## scene setup ##########################
def setup_scene(show_viewer: bool = True):
    """Create and configure the simulation scene.
    
    Args:
        show_viewer: Whether to display the viewer
    """
    scene = gs.Scene(
        sim_options = gs.options.SimOptions(dt = 0.01),
        viewer_options = gs.options.ViewerOptions(
            camera_pos    = (3, -1, 1.5),
            camera_lookat = (0.0, 0.0, 0.5),
            camera_fov    = 30,
            max_FPS       = 60,
        ),
        show_viewer = show_viewer,
    )
    return scene

def setup_entities(scene):
    """Add entities to the scene."""
    plane = scene.add_entity(gs.morphs.Plane())
    cube = scene.add_entity(
        gs.morphs.Box(
            size = (0.04, 0.04, 0.04),
            pos  = (0.65, 0.0, 0.02),
        )
    )
    franka = scene.add_entity(
        gs.morphs.MJCF(file='xml/franka_emika_panda/panda.xml'),
    )
    return plane, cube, franka

def setup_camera(scene):
    """Add wrist-mounted camera to the scene."""
    cam = scene.add_camera(
        res    = (640, 480),
        pos    = (0.6, 0.0, 0.3),
        lookat = (0.7, 0.0, 0.3),
        fov    = 60,
        GUI    = False,
    )
    return cam

def configure_robot(franka):
    """Set control gains and force limits for the robot."""
    franka.set_dofs_kp(
        np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100]),
    )
    franka.set_dofs_kv(
        np.array([450, 450, 350, 350, 200, 200, 200, 10, 10]),
    )
    franka.set_dofs_force_range(
        np.array([-87, -87, -87, -87, -12, -12, -12, -100, -100]),
        np.array([ 87,  87,  87,  87,  12,  12,  12,  100,  100]),
    )

########################## camera functions ##########################
def update_wrist_camera(cam, end_effector):
    """
    Update the camera pose so it stays mounted on the hand link.
    
    Args:
        cam: Camera object (or None if disabled)
        end_effector: End-effector link
    """
    if cam is None:
        return
    
    import torch
    
    hand_pos = end_effector.get_pos()
    if isinstance(hand_pos, torch.Tensor):
        hand_pos = hand_pos.cpu().numpy()
    
    cam_offset = np.array([0.15, 0.0, 0.05])
    cam_pos = hand_pos + cam_offset
    cam_lookat = hand_pos + np.array([0.0, 0.0, -0.05])
    
    cam.set_pose(
        pos    = tuple(cam_pos),
        lookat = tuple(cam_lookat),
    )

########################## sensing functions ##########################
def get_proprioception(franka, end_effector) -> Dict[str, np.ndarray]:
    """
    Get robot proprioception data (joint states, EE pose/twist, gripper).
    
    Returns dict with:
    - q: joint positions (9,) [7 arm joints + 2 gripper]
    - dq: joint velocities (9,)
    - tau: joint torques/efforts (9,)
    - ee_pos: end-effector position (3,)
    - ee_quat: end-effector orientation as quaternion (4,)
    - ee_lin_vel: end-effector linear velocity (3,)
    - gripper_width: distance between gripper fingers (scalar)
    """
    q = franka.get_qpos().cpu().numpy()
    dq = franka.get_dofs_velocity().cpu().numpy()
    tau = franka.get_dofs_force().cpu().numpy()
    
    ee_pos = end_effector.get_pos().cpu().numpy()
    ee_quat = end_effector.get_quat().cpu().numpy()
    ee_lin_vel = end_effector.get_vel().cpu().numpy()
    
    gripper_width = q[7] + q[8]
    
    return {
        'q': q,
        'dq': dq,
        'tau': tau,
        'ee_pos': ee_pos,
        'ee_quat': ee_quat,
        'ee_lin_vel': ee_lin_vel,
        'gripper_width': gripper_width,
    }

def detect_contact_with_object(franka, end_effector, cube) -> Dict[str, Any]:
    """
    Detect contact between gripper and cube using ground-truth physics info.
    
    This is the "cheat" sensor for auto-labeling in simulation.
    On real robot, you'd use force/torque sensors + tactile sensors.
    
    Returns dict with:
    - in_contact: bool, whether gripper is touching cube
    - contact_force: float, magnitude of gripper force
    - cube_ee_distance: float, distance between cube and EE
    - cube_lifted: bool, whether cube is lifted off ground
    """
    try:
        cube_pos = cube.get_pos().cpu().numpy()
        ee_pos = end_effector.get_pos().cpu().numpy()
        
        gripper_qpos = franka.get_qpos().cpu().numpy()[7:9]
        gripper_forces = franka.get_dofs_force().cpu().numpy()[7:9]
        gripper_width = gripper_qpos.sum()
        
        distance = np.linalg.norm(cube_pos - ee_pos)
        cube_lifted = cube_pos[2] > 0.025
        
        close_and_closed = (distance < 0.10) and (gripper_width < 0.055)
        
        gripper_dvel = franka.get_dofs_velocity().cpu().numpy()[7:9]
        gripper_stable = (gripper_width < 0.055) and (np.abs(gripper_dvel).max() < 0.01)
        
        in_contact = cube_lifted or (close_and_closed and gripper_stable)
        contact_force = np.abs(gripper_forces).max()
        
        return {
            'in_contact': bool(in_contact),
            'contact_force': float(contact_force),
            'num_contact_points': 2 if in_contact else 0,
            'cube_ee_distance': float(distance),
            'cube_lifted': bool(cube_lifted),
            'gripper_width': float(gripper_width),
        }
    except Exception as e:
        print(f"Contact detection error: {e}")
        return {
            'in_contact': False,
            'contact_force': 0.0,
            'num_contact_points': 0,
            'cube_ee_distance': 0.0,
            'cube_lifted': False,
            'gripper_width': 0.0,
        }

def get_object_state(cube) -> Dict[str, np.ndarray]:
    """
    Get cube state for additional labeling (dropped, slipped, etc.).
    
    Returns dict with:
    - obj_pos: cube position (3,)
    - obj_quat: cube orientation (4,)
    - obj_lin_vel: cube linear velocity (3,)
    """
    obj_pos = cube.get_pos().cpu().numpy()
    obj_quat = cube.get_quat().cpu().numpy()
    obj_lin_vel = cube.get_vel().cpu().numpy()
    
    return {
        'obj_pos': obj_pos,
        'obj_quat': obj_quat,
        'obj_lin_vel': obj_lin_vel,
    }

def collect_sensor_data(franka, end_effector, cube, cam, include_vision: bool = False) -> Dict[str, Any]:
    """
    Collect all sensor data for current timestep.
    
    Args:
        franka: Robot entity
        end_effector: End-effector link
        cube: Cube entity
        cam: Camera object
        include_vision: Whether to capture and include RGBD images
    
    Returns:
        Dict with all sensor readings
    """
    data = {}
    
    proprio = get_proprioception(franka, end_effector)
    data.update(proprio)
    
    contact = detect_contact_with_object(franka, end_effector, cube)
    data.update(contact)
    
    obj_state = get_object_state(cube)
    data.update(obj_state)
    
    if include_vision:
        rgb, depth, seg, normal = cam.render(depth=True)
        data['rgb'] = rgb
        data['depth'] = depth
    
    return data

########################## motion execution functions ##########################
def execute_trajectory(franka, scene, cam, end_effector, cube, logger, 
                       path, display_video=True, check_contact=True):
    """
    Execute a planned trajectory with sensor data collection.
    
    Args:
        franka: Robot entity
        scene: Simulation scene
        cam: Camera object
        end_effector: End-effector link
        cube: Cube entity
        logger: SensorDataLogger instance
        path: Trajectory waypoints
        display_video: Whether to display RGB/Depth video
        check_contact: Whether to check for contact lost events
    """
    for waypoint in path:
        franka.control_dofs_position(waypoint)
        scene.step()
        update_wrist_camera(cam, end_effector)
        
        sensor_data = collect_sensor_data(franka, end_effector, cube, cam, include_vision=False)
        logger.log_step(sensor_data)
        
        if check_contact and logger.detect_contact_lost():
            print(f"CONTACT LOST at timestep {logger.timestep}")
        
        if display_video and cam is not None:
            rgb, depth, seg, normal = cam.render(depth=True)
            depth_vis = (depth / depth.max() * 255).astype('uint8')
            cv2.imshow("RGB", rgb[:, :, ::-1])
            cv2.imshow("Depth", depth_vis)
            cv2.waitKey(1)

def execute_steps(franka, scene, cam, end_effector, cube, logger, 
                 num_steps, motors_dof=None, qpos=None, finger_force=None, fingers_dof=None,
                 display_video=True, print_status=False, print_interval=20, phase_name=""):
    """
    Execute simulation steps with sensor data collection.
    
    Args:
        franka: Robot entity
        scene: Simulation scene
        cam: Camera object
        end_effector: End-effector link
        cube: Cube entity
        logger: SensorDataLogger instance
        num_steps: Number of steps to execute
        motors_dof: Motor DOF indices (for position control)
        qpos: Target joint positions
        finger_force: Finger force command (for force control)
        fingers_dof: Finger DOF indices
        display_video: Whether to display RGB/Depth video
        print_status: Whether to print status updates
        print_interval: Steps between status prints
        phase_name: Name of current phase for logging
    """
    if phase_name:
        print(f"{phase_name}...")
    
    if qpos is not None and motors_dof is not None:
        franka.control_dofs_position(qpos, motors_dof)
    
    if finger_force is not None and fingers_dof is not None:
        franka.control_dofs_force(finger_force, fingers_dof)
    
    for i in range(num_steps):
        scene.step()
        update_wrist_camera(cam, end_effector)
        
        sensor_data = collect_sensor_data(franka, end_effector, cube, cam, include_vision=False)
        logger.log_step(sensor_data)
        
        if print_status and i % print_interval == 0:
            contact_str = "IN CONTACT" if sensor_data['in_contact'] else "NO CONTACT"
            cube_height = sensor_data['obj_pos'][2]
            print(f"  Step {i}: {contact_str}, Force: {sensor_data['contact_force']:.3f}N, "
                  f"Gripper: {sensor_data['gripper_width']:.4f}m, Cube Z: {cube_height:.3f}m, "
                  f"Lifted: {sensor_data['cube_lifted']}")
        
        if logger.detect_contact_lost():
            print(f"CONTACT LOST at timestep {logger.timestep}, Cube height: {sensor_data['obj_pos'][2]:.3f}m")
        
        if display_video and cam is not None:
            rgb, depth, seg, normal = cam.render(depth=True)
            depth_vis = (depth / depth.max() * 255).astype('uint8')
            cv2.imshow("RGB", rgb[:, :, ::-1])
            cv2.imshow("Depth", depth_vis)
            cv2.waitKey(1)

def run_pick_and_place_demo(franka, scene, cam, end_effector, cube, logger, motors_dof, fingers_dof,
                           display_video=True):
    """
    Execute a complete pick-and-place demonstration.
    
    Args:
        franka: Robot entity
        scene: Simulation scene
        cam: Camera object (or None if disabled)
        end_effector: End-effector link
        cube: Cube entity
        logger: SensorDataLogger instance
        motors_dof: Motor DOF indices
        fingers_dof: Finger DOF indices
        display_video: Whether to display camera playback
    """
    # Move to pre-grasp pose
    qpos = franka.inverse_kinematics(
        link = end_effector,
        pos  = np.array([0.65, 0.0, 0.25]),
        quat = np.array([0, 1, 0, 0]),
    )
    qpos[-2:] = 0.04  # Open gripper
    path = franka.plan_path(qpos_goal=qpos, num_waypoints=200)
    
    execute_trajectory(franka, scene, cam, end_effector, cube, logger, path, display_video=display_video)
    
    # Stabilize at pre-grasp
    execute_steps(franka, scene, cam, end_effector, cube, logger, 
                 num_steps=100, display_video=display_video)
    
    # Reach to cube
    qpos = franka.inverse_kinematics(
        link = end_effector,
        pos  = np.array([0.65, 0.0, 0.130]),
        quat = np.array([0, 1, 0, 0]),
    )
    execute_steps(franka, scene, cam, end_effector, cube, logger, 
                 num_steps=100, motors_dof=motors_dof, qpos=qpos[:-2], display_video=display_video)
    
    # Grasp
    execute_steps(franka, scene, cam, end_effector, cube, logger, 
                 num_steps=100, motors_dof=motors_dof, qpos=qpos[:-2],
                 finger_force=np.array([-0.5, -0.5]), fingers_dof=fingers_dof,
                 print_status=True, print_interval=20, phase_name="Grasping", display_video=display_video)
    
    # Lift
    qpos = franka.inverse_kinematics(
        link=end_effector,
        pos=np.array([0.65, 0.0, 0.28]),
        quat=np.array([0, 1, 0, 0]),
    )
    execute_steps(franka, scene, cam, end_effector, cube, logger, 
                 num_steps=200, motors_dof=motors_dof, qpos=qpos[:-2],
                 print_status=True, print_interval=40, phase_name="Lifting", display_video=display_video)

########################## main ##########################
def main():
    """Main execution function."""
    # Determine graphics mode
    show_viewer = not args.headless
    show_camera_playback = not (args.headless or args.sim_only)
    
    # Initialize logger
    logger = SensorDataLogger()
    
    # Setup scene
    scene = setup_scene(show_viewer=show_viewer)
    plane, cube, franka = setup_entities(scene)
    cam = setup_camera(scene) if show_camera_playback else None
    
    # Build scene
    scene.build()
    
    # Configure robot
    motors_dof = np.arange(7)
    fingers_dof = np.arange(7, 9)
    configure_robot(franka)
    
    time.sleep(2.0)
    
    end_effector = franka.get_link('hand')
    if cam is not None:
        update_wrist_camera(cam, end_effector)
    
    # Run demonstration
    run_pick_and_place_demo(franka, scene, cam, end_effector, cube, logger, motors_dof, fingers_dof,
                           display_video=show_camera_playback)
    
    # Save and report results
    logger.save('sensor_data.npz')
    stats = logger.get_summary_stats()
    print(f"\nCollected {stats['total_timesteps']} timesteps of sensor data")
    print(f"Total contact events: {stats['total_contact']}")
    print(f"Contact lost events: {stats['contact_lost_events']}")

if __name__ == "__main__":
    main()
