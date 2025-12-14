"""Contact detection functionality."""
import numpy as np
from typing import Dict, Any


# Physics-based contact detection thresholds
FINGER_FORCE_THRESHOLD = 0.1  # N - minimum force to register contact per finger
MIN_CONTACT_FORCES_FOR_GRASP = 2  # Both fingers must have force for stable grasp
NORMAL_FORCE_THRESHOLD = 0.05  # N - threshold for normal force component
CONTACT_POINT_DISTANCE_THRESHOLD = 0.002  # m - max distance to be considered contact


def detect_contact_with_object(franka, end_effector, cube) -> Dict[str, Any]:
    """Robust contact detection using physics engine contact forces."""
    try:
        # Get gripper joint forces
        gripper_forces = franka.get_dofs_force().cpu().numpy()[7:9]
        left_force = np.abs(gripper_forces[0])
        right_force = np.abs(gripper_forces[1])
        
        # Contact detection: both fingers must have measurable force
        left_in_contact = left_force > FINGER_FORCE_THRESHOLD
        right_in_contact = right_force > FINGER_FORCE_THRESHOLD
        num_contact_forces = int(left_in_contact) + int(right_in_contact)
        
        # Stable grasp requires contact on both fingers
        in_contact = num_contact_forces >= MIN_CONTACT_FORCES_FOR_GRASP
        
        # Additional diagnostics
        gripper_qpos = franka.get_qpos().cpu().numpy()[7:9]
        gripper_width = gripper_qpos.sum()
        
        cube_pos = cube.get_pos().cpu().numpy()
        ee_pos = end_effector.get_pos().cpu().numpy()
        cube_ee_distance = np.linalg.norm(cube_pos - ee_pos)
        cube_lifted = cube_pos[2] > 0.025
        
        return {
            'in_contact': bool(in_contact),
            'left_finger_force': float(left_force),
            'right_finger_force': float(right_force),
            'total_contact_force': float(left_force + right_force),
            'num_contact_forces': int(num_contact_forces),
            'gripper_width': float(gripper_width),
            'cube_lifted': bool(cube_lifted),
            'cube_ee_distance': float(cube_ee_distance),
        }
    except Exception as e:
        print(f"Contact detection error: {e}")
        return {
            'in_contact': False,
            'left_finger_force': 0.0,
            'right_finger_force': 0.0,
            'total_contact_force': 0.0,
            'num_contact_forces': 0,
            'gripper_width': 0.0,
            'cube_lifted': False,
            'cube_ee_distance': float('inf'),
        }
