import numpy as np
import genesis as gs
import time
import cv2 

########################## init ##########################
gs.init(backend=gs.gpu)

########################## create a scene ##########################
scene = gs.Scene(
    sim_options = gs.options.SimOptions(
        dt = 0.01,
    ),
    viewer_options = gs.options.ViewerOptions(
        camera_pos    = (3, -1, 1.5),
        camera_lookat = (0.0, 0.0, 0.5),
        camera_fov    = 30,
        max_FPS       = 60,
    ),
    show_viewer = True,
)

########################## entities ##########################
plane = scene.add_entity(
    gs.morphs.Plane(),
)
cube = scene.add_entity(
    gs.morphs.Box(
        size = (0.04, 0.04, 0.04),
        pos  = (0.65, 0.0, 0.02),
    )
)
franka = scene.add_entity(
    gs.morphs.MJCF(file='xml/franka_emika_panda/panda.xml'),
)

########################## wrist camera ##########################
### >>> NEW: add a camera that we will “mount” on the hand
cam = scene.add_camera(
    res    = (640, 480),
    pos    = (0.6, 0.0, 0.3),    # temporary initial pose
    lookat = (0.7, 0.0, 0.3),
    fov    = 60,
    GUI    = False,              # set True if you want an OpenCV window
)

########################## build ##########################
scene.build()

motors_dof = np.arange(7)
fingers_dof = np.arange(7, 9)

# set control gains
franka.set_dofs_kp(
    np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100]),
)
franka.set_dofs_kv(
    np.array([450, 450, 350, 350, 200, 200, 200, 10, 10]),
)
franka.set_dofs_force_range(
    np.array([-87, -87, -87, -87, -12, -12, -12, -100, -100]),
    np.array([ 87,  87,  87,  87,  12,  12,  12,  100,  100]),
)

time.sleep(2.0)  # wait a bit to let the viewer pop up

# get the end-effector link
end_effector = franka.get_link('hand')

########################## camera update helper ##########################
### >>> NEW: function to keep camera rigidly attached to the hand
def update_wrist_camera():
    """
    Update the camera pose so it stays mounted on the hand link.

    - Camera position = hand position + small forward offset
    - Camera lookat   = a point slightly in front of the hand
    """
    import torch

    # link.get_pos() / get_quat() return torch tensors in world frame
    hand_pos = end_effector.get_pos()          # shape (3,)
    if isinstance(hand_pos, torch.Tensor):
        hand_pos = hand_pos.cpu().numpy()

    # simple fixed offset in world frame (you can tune this)
    cam_offset = np.array([0.15, 0.0, 0.05])    # 15 cm in +X, 5 cm in +Z (above and away from the hand)
    cam_pos = hand_pos + cam_offset

    # look down at the gripper fingers
    cam_lookat = hand_pos + np.array([0.0, 0.0, -0.05])

    cam.set_pose(
        pos    = tuple(cam_pos),
        lookat = tuple(cam_lookat),
    )

# initialize camera pose once
update_wrist_camera()

########################## motion ##########################
# move to pre-grasp pose
qpos = franka.inverse_kinematics(
    link = end_effector,
    pos  = np.array([0.65, 0.0, 0.25]),
    quat = np.array([0, 1, 0, 0]),
)
# gripper open pos
qpos[-2:] = 0.04
path = franka.plan_path(
    qpos_goal     = qpos,
    num_waypoints = 200, # 2s duration
)

# execute the planned path
for waypoint in path:
    franka.control_dofs_position(waypoint)
    scene.step()
    update_wrist_camera()
    # >>> LIVE VIEW HERE
    rgb, depth, seg, normal = cam.render(depth=True)
    depth_vis = (depth / depth.max() * 255).astype('uint8')

    cv2.imshow("RGB", rgb[:, :, ::-1])
    cv2.imshow("Depth", depth_vis)
    cv2.waitKey(1)


# allow robot to reach the last waypoint
for i in range(100):
    scene.step()
    update_wrist_camera()
    # rgb, depth = cam.render(depth=True)

# reach
qpos = franka.inverse_kinematics(
    link = end_effector,
    pos  = np.array([0.65, 0.0, 0.130]),
    quat = np.array([0, 1, 0, 0]),
)
franka.control_dofs_position(qpos[:-2], motors_dof)
for i in range(100):
    scene.step()
    update_wrist_camera()
    # >>> LIVE VIEW HERE
    rgb, depth, seg, normal = cam.render(depth=True)
    depth_vis = (depth / depth.max() * 255).astype('uint8')

    cv2.imshow("RGB", rgb[:, :, ::-1])
    cv2.imshow("Depth", depth_vis)
    cv2.waitKey(1)

# grasp
franka.control_dofs_position(qpos[:-2], motors_dof)
franka.control_dofs_force(np.array([-0.5, -0.5]), fingers_dof)

for i in range(100):
    scene.step()
    update_wrist_camera()
    # >>> LIVE VIEW HERE
    rgb, depth, seg, normal = cam.render(depth=True)
    depth_vis = (depth / depth.max() * 255).astype('uint8')

    cv2.imshow("RGB", rgb[:, :, ::-1])
    cv2.imshow("Depth", depth_vis)
    cv2.waitKey(1)

# lift
qpos = franka.inverse_kinematics(
    link=end_effector,
    pos=np.array([0.65, 0.0, 0.28]),
    quat=np.array([0, 1, 0, 0]),
)
franka.control_dofs_position(qpos[:-2], motors_dof)
for i in range(200):
    scene.step()
    update_wrist_camera()
    # >>> LIVE VIEW HERE
    rgb, depth, seg, normal = cam.render(depth=True)
    depth_vis = (depth / depth.max() * 255).astype('uint8')

    cv2.imshow("RGB", rgb[:, :, ::-1])
    cv2.imshow("Depth", depth_vis)
    cv2.waitKey(1)
