"""Trajectory execution functionality."""
import cv2
from src.camera import update_wrist_camera
from src.sensors import collect_sensor_data


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
