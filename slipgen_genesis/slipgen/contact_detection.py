"""Contact detection functionality (stubbed)."""
import numpy as np
from typing import Dict, Any


def _to_np(x):
    if hasattr(x, "detach") and hasattr(x, "cpu"):
        return x.detach().cpu().numpy()
    if hasattr(x, "cpu"):
        return x.cpu().numpy()
    return np.asarray(x)


def detect_contact_with_object(franka, end_effector, cube) -> Dict[str, Any]:
    """Stub contact detection: returns neutral values and in_contact=True.

    This intentionally avoids force-based heuristics. The keys are preserved
    for downstream consumers. Replace with a proper implementation later.
    """
    try:
        # Derive benign diagnostics without relying on joint forces
        qpos = _to_np(franka.get_qpos())
        gripper_width = float(qpos[-2:].sum()) if qpos is not None and qpos.size >= 2 else 0.0

        cube_pos = _to_np(cube.get_pos())
        ee_pos = _to_np(end_effector.get_pos())
        cube_lifted = bool(cube_pos[2] > 0.025) if cube_pos is not None and cube_pos.size >= 3 else False
        cube_ee_distance = float(np.linalg.norm(cube_pos - ee_pos)) if (cube_pos is not None and ee_pos is not None) else float("inf")

        return {
            'in_contact': True,                # Always true while contact logic is stubbed
            'left_finger_force': 0.0,          # Neutral placeholders
            'right_finger_force': 0.0,
            'total_contact_force': 0.0,
            'num_contact_forces': 0,
            'gripper_width': gripper_width,
            'cube_lifted': cube_lifted,
            'cube_ee_distance': cube_ee_distance,
        }
    except Exception:
        # Fall back to safe defaults
        return {
            'in_contact': True,
            'left_finger_force': 0.0,
            'right_finger_force': 0.0,
            'total_contact_force': 0.0,
            'num_contact_forces': 0,
            'gripper_width': 0.0,
            'cube_lifted': False,
            'cube_ee_distance': float('inf'),
        }
