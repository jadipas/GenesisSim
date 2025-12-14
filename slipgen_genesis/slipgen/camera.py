"""Wrist-mounted camera control."""
import numpy as np
import genesis.utils.geom as gu


def mount_wrist_camera(cam, hand_link, offset_pos=None, offset_quat=None):
    """Attach the camera to the hand link using a local-frame transform."""
    if cam is None or hand_link is None:
        return

    pos = np.array(offset_pos if offset_pos is not None else [0.15, 0.0, 0.0], dtype=np.float32)
    # Rotate 135 degrees around Y to look at fingers at an angle: [0, 0.9239, 0, 0.3827]
    quat = np.array(offset_quat if offset_quat is not None else [0.0, 0.92387953, 0.0, 0.38268343], dtype=np.float32)

    offset_T = gu.trans_quat_to_T(pos, quat)
    cam.attach(hand_link, offset_T)
    cam.move_to_attach()


def update_wrist_camera(cam, _end_effector=None):
    """Move an attached camera to follow its parent link."""
    if cam is None:
        return

    cam.move_to_attach()
