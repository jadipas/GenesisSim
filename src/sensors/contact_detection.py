"""Contact detection functionality."""
import numpy as np
from typing import Dict, Any


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
