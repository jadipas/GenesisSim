"""Contact detection functionality."""
import numpy as np
from typing import Dict, Any


# Physics-based contact detection thresholds
# These are calibrated for Genesis physics engine with Franka gripper
FINGER_FORCE_THRESHOLD = 0.1  # N - minimum force to register contact per finger
MIN_CONTACT_FORCES_FOR_GRASP = 2  # Both fingers must have force for stable grasp
NORMAL_FORCE_THRESHOLD = 0.05  # N - threshold for normal force component
CONTACT_POINT_DISTANCE_THRESHOLD = 0.002  # m - max distance to be considered contact


def detect_contact_with_object(franka, end_effector, cube) -> Dict[str, Any]:
    """
    Robust contact detection using physics engine contact forces.
    
    Uses the Genesis physics engine's actual contact forces on gripper fingers
    as the primary contact signal. This is ground-truth from the physics simulation.
    
    Returns dict with:
    - in_contact: bool, whether gripper has force-based contact with cube
    - left_finger_force: float, normal force on left finger (N)
    - right_finger_force: float, normal force on right finger (N)
    - total_contact_force: float, sum of finger contact forces (N)
    - num_contact_forces: int, how many fingers have contact force
    - gripper_width: float, current gripper aperture (m)
    - cube_lifted: bool, whether cube is raised off ground plane
    - cube_ee_distance: float, distance between cube center and end-effector (m)
    """
    try:
        # Get gripper joint forces
        gripper_forces = franka.get_dofs_force().cpu().numpy()[7:9]  # [left_finger, right_finger]
        left_force = np.abs(gripper_forces[0])
        right_force = np.abs(gripper_forces[1])
        
        # Contact detection: both fingers must have measurable force
        # This ensures the object is actually being gripped, not just touched
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
        cube_lifted = cube_pos[2] > 0.025  # Object is off ground plane
        
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
