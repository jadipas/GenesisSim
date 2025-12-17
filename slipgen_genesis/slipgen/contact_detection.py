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
DEFAULT_SLIP_THRESHOLD = 0.01


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
    """Detect contact with object using distance and gripper state.
    
    Uses a distance-based approach since Genesis joint forces from get_dofs_force()
    may not reliably reflect contact forces when using position control.
    
    Contact is determined by:
    1. Small distance between cube center and end-effector
    2. Gripper being partially closed (not fully open)
    """
    try:
        # Get joint states and forces
        qpos = _to_np(franka.get_qpos())
        tau = _to_np(franka.get_dofs_force())  # Joint torques/forces [N·m for arm, N for fingers]
        
        # Gripper state - sum of both finger positions
        # Fully open: ~0.08 (0.04 + 0.04), Fully closed: ~0.0 or negative
        gripper_width = float(qpos[-2:].sum()) if qpos is not None and qpos.size >= 2 else 0.0
        
        # Extract fingertip forces (DOFs 7 and 8 are the two gripper fingers)
        # Note: These may not be reliable for contact detection in position control mode
        left_finger_force = float(np.abs(tau[7])) if tau is not None and tau.size > 7 else 0.0
        right_finger_force = float(np.abs(tau[8])) if tau is not None and tau.size > 8 else 0.0
        total_contact_force = left_finger_force + right_finger_force
        
        # Object state
        cube_pos = _to_np(cube.get_pos())
        ee_pos = _to_np(end_effector.get_pos())
        cube_lifted = bool(cube_pos[2] > 0.025) if cube_pos is not None and cube_pos.size >= 3 else False
        cube_ee_distance = float(np.linalg.norm(cube_pos - ee_pos)) if (cube_pos is not None and ee_pos is not None) else float("inf")

        # Distance-based contact detection (more reliable than force-based)
        # Contact if: cube is close to EE AND gripper is not fully open
        distance_threshold = 0.12  # meters - cube center to EE distance when grasped
        gripper_open_threshold = 0.06  # Gripper width when "open" (fully open is ~0.08)
        
        gripper_closing = gripper_width < gripper_open_threshold
        cube_close = cube_ee_distance < distance_threshold
        in_contact = gripper_closing and cube_close
        
        # For backward compatibility, count how many fingers show force
        contact_threshold = 0.1  # Newtons
        num_contact_forces = sum([
            left_finger_force > contact_threshold,
            right_finger_force > contact_threshold
        ])

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
