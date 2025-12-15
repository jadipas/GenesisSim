"""Contact detection functionality (stubbed)."""
import numpy as np
from typing import Dict, Any, Optional


def _to_np(x):
    if hasattr(x, "detach") and hasattr(x, "cpu"):
        return x.detach().cpu().numpy()
    if hasattr(x, "cpu"):
        return x.cpu().numpy()
    return np.asarray(x)


# Default slip detection threshold in meters (5cm)
DEFAULT_SLIP_THRESHOLD = 0.05


def detect_slip_by_distance(
    cube_pos: np.ndarray,
    ee_pos: np.ndarray,
    baseline_distance: float,
    threshold: float = DEFAULT_SLIP_THRESHOLD,
) -> Dict[str, Any]:
    """Detect slip based on relative distance change between cube and gripper.
    
    This is more reliable than force-based detection in simulation environments.
    Slip is detected when the cube-to-end-effector distance deviates from the
    baseline distance captured at grasp by more than the threshold.
    
    Args:
        cube_pos: Current cube position [x, y, z] in world frame
        ee_pos: Current end-effector position [x, y, z] in world frame
        baseline_distance: Distance captured at initial secure grasp (meters)
        threshold: Max allowed displacement before slip is flagged (meters)
                   Default: 0.015m (1.5cm) - tune based on cube size
                   Conservative: 0.005m (5mm) - catches minor slippage
                   Permissive: 0.020m+ (2cm+) - only catches significant drops
    
    Returns:
        dict with slip detection results:
            - slip_detected: bool, True if slip threshold exceeded
            - displacement_from_baseline: float, absolute distance change (m)
            - current_ee_cube_distance: float, current distance (m)
            - slip_threshold: float, threshold used (m)
            - vertical_slip: float, vertical displacement component (m)
            - horizontal_slip: float, horizontal displacement component (m)
    """
    cube_pos = _to_np(cube_pos)
    ee_pos = _to_np(ee_pos)
    
    current_distance = float(np.linalg.norm(cube_pos - ee_pos))
    displacement = abs(current_distance - baseline_distance)
    
    # Decompose into vertical and horizontal components for diagnostics
    relative_pos = cube_pos - ee_pos
    vertical_component = abs(relative_pos[2])  # Z-axis
    horizontal_component = float(np.linalg.norm(relative_pos[:2]))  # XY-plane
    
    return {
        'slip_detected': displacement > threshold,
        'displacement_from_baseline': displacement,
        'current_ee_cube_distance': current_distance,
        'slip_threshold': threshold,
        'vertical_slip': vertical_component,
        'horizontal_slip': horizontal_component,
    }


def detect_contact_with_object(franka, end_effector, cube) -> Dict[str, Any]:
    """Detect contact with object using gripper force measurement.
    
    Measures forces at left (DOF 7) and right (DOF 8) finger joints.
    Contact is determined by non-zero finger forces.
    """
    try:
        # Get joint states and forces
        qpos = _to_np(franka.get_qpos())
        tau = _to_np(franka.get_dofs_force())  # Joint torques/forces [N·m for arm, N for fingers]
        
        # Gripper state
        gripper_width = float(qpos[-2:].sum()) if qpos is not None and qpos.size >= 2 else 0.0
        
        # Extract fingertip forces (DOFs 7 and 8 are the two gripper fingers)
        # Forces are measured in Newtons at the fingertips
        left_finger_force = float(np.abs(tau[7])) if tau is not None and tau.size > 7 else 0.0
        right_finger_force = float(np.abs(tau[8])) if tau is not None and tau.size > 8 else 0.0
        total_contact_force = left_finger_force + right_finger_force
        
        # Contact detection: gripper is in contact if there's measurable force on fingers
        contact_threshold = 0.1  # Newtons - minimum force to register contact
        num_contact_forces = sum([
            left_finger_force > contact_threshold,
            right_finger_force > contact_threshold
        ])
        in_contact = num_contact_forces >= 1
        
        # Object state
        cube_pos = _to_np(cube.get_pos())
        ee_pos = _to_np(end_effector.get_pos())
        cube_lifted = bool(cube_pos[2] > 0.025) if cube_pos is not None and cube_pos.size >= 3 else False
        cube_ee_distance = float(np.linalg.norm(cube_pos - ee_pos)) if (cube_pos is not None and ee_pos is not None) else float("inf")

        return {
            'in_contact': in_contact,
            'left_finger_force': left_finger_force,
            'right_finger_force': right_finger_force,
            'total_contact_force': total_contact_force,
            'num_contact_forces': num_contact_forces,
            'gripper_width': gripper_width,
            'cube_lifted': cube_lifted,
            'cube_ee_distance': cube_ee_distance,
        }
    except Exception as e:
        # Fall back to safe defaults with debug message
        print(f"[Contact Detection Error] {e}")
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
