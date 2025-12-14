"""Trajectory APIs with optional disturbance injection."""
from typing import Iterable, Sequence, Optional, Dict, Any
import numpy as np

from slipgen.trajectory import generate_composite_trajectory as _generate_composite_trajectory
from slipgen.trajectory import execute_trajectory as _execute_trajectory


def generate_composite(franka, end_effector, waypoints: Iterable[dict], default_steps: int = 150,
                        finger_qpos: Optional[float | Sequence[float]] = None):
    return _generate_composite_trajectory(franka, end_effector, waypoints, default_steps, finger_qpos)


def _apply_accel_bump(path: np.ndarray, level: int) -> np.ndarray:
    """Compress mid-segment samples to emulate an acceleration burst."""
    if level <= 0 or len(path) < 10:
        return path
    n = len(path)
    mid = n // 2
    window = max(5, n // 6)
    severity = {1: 0.5, 2: 0.35, 3: 0.2}.get(level, 0.5)
    start = max(0, mid - window // 2)
    end = min(n, mid + window // 2)

    pre = path[:start]
    bump = path[start:end]
    post = path[end:]

    idx = np.linspace(0, len(bump) - 1, max(2, int(len(bump) * severity))).astype(int)
    bump_fast = bump[idx]

    return np.vstack([pre, bump_fast, post])


def execute(franka, scene, cam, end_effector, cube, logger, path,
            display_video: bool = True, check_contact: bool = True,
            step_callbacks: Optional[Dict[int, Any]] = None, phase_name: str = "",
            debug: bool = False, disturb_level: int = 0):
    path2 = _apply_accel_bump(np.asarray(path), disturb_level)
    return _execute_trajectory(franka, scene, cam, end_effector, cube, logger, path2,
                               display_video, check_contact, step_callbacks, phase_name, debug)
