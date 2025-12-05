"""Object state sensing functionality."""
import numpy as np
from typing import Dict


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
