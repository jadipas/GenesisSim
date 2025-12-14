"""Wrist-mounted camera control."""
import numpy as np
import torch


def update_wrist_camera(cam, end_effector):
    """Update the camera pose so it stays mounted on the hand link."""
    if cam is None:
        return
    
    hand_pos = end_effector.get_pos()
    if isinstance(hand_pos, torch.Tensor):
        hand_pos = hand_pos.cpu().numpy()
    
    cam_offset = np.array([0.15, 0.0, 0.05])
    cam_pos = hand_pos + cam_offset
    cam_lookat = hand_pos + np.array([0.0, 0.0, -0.05])
    
    cam.set_pose(
        pos    = tuple(cam_pos),
        lookat = tuple(cam_lookat),
    )
