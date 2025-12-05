"""Step-by-step execution functionality."""
import cv2
from src.camera import update_wrist_camera
from src.sensors import collect_sensor_data


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
