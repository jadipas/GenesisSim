"""Proprioceptive sensing functionality."""
import numpy as np
from typing import Dict


def get_proprioception(franka, end_effector) -> Dict[str, np.ndarray]:
    """
    Get robot proprioception data (joint states, EE pose/twist, gripper).
    
    Returns dict with:
    - q: joint positions (9,) [7 arm joints + 2 gripper]
    - dq: joint velocities (9,)
    - tau: joint torques/efforts (9,)
    - ee_pos: end-effector position (3,)
    - ee_quat: end-effector orientation as quaternion (4,)
    - ee_lin_vel: end-effector linear velocity (3,)
    - gripper_width: distance between gripper fingers (scalar)
    """
    q = franka.get_qpos().cpu().numpy()
    dq = franka.get_dofs_velocity().cpu().numpy()
    tau = franka.get_dofs_force().cpu().numpy()
    
    ee_pos = end_effector.get_pos().cpu().numpy()
    ee_quat = end_effector.get_quat().cpu().numpy()
    ee_lin_vel = end_effector.get_vel().cpu().numpy()
    
    gripper_width = q[7] + q[8]
    
    return {
        'q': q,
        'dq': dq,
        'tau': tau,
        'ee_pos': ee_pos,
        'ee_quat': ee_quat,
        'ee_lin_vel': ee_lin_vel,
        'gripper_width': gripper_width,
    }
