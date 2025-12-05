"""Pick-and-place demonstration functionality."""
import numpy as np
from .trajectory import execute_trajectory, generate_composite_trajectory
from .steps import execute_steps


def run_pick_and_place_demo(franka, scene, cam, end_effector, cube, logger, motors_dof, fingers_dof,
                           display_video=True):
    """
    Execute a complete pick-and-place demonstration.
    
    Args:
        franka: Robot entity
        scene: Simulation scene
        cam: Camera object (or None if disabled)
        end_effector: End-effector link
        cube: Cube entity
        logger: SensorDataLogger instance
        motors_dof: Motor DOF indices
        fingers_dof: Finger DOF indices
        display_video: Whether to display camera playback
    """
    # Move to pre-grasp pose
    qpos = franka.inverse_kinematics(
        link = end_effector,
        pos  = np.array([0.65, 0.0, 0.25]),
        quat = np.array([0, 1, 0, 0]),
    )
    qpos[-2:] = 0.04  # Open gripper
    path = franka.plan_path(qpos_goal=qpos, num_waypoints=200)
    
    execute_trajectory(franka, scene, cam, end_effector, cube, logger, path, display_video=display_video)
    
    # Stabilize at pre-grasp
    execute_steps(franka, scene, cam, end_effector, cube, logger, 
                 num_steps=100, display_video=display_video)
    
    # Reach to cube
    qpos = franka.inverse_kinematics(
        link = end_effector,
        pos  = np.array([0.65, 0.0, 0.130]),
        quat = np.array([0, 1, 0, 0]),
    )
    execute_steps(franka, scene, cam, end_effector, cube, logger, 
                 num_steps=100, motors_dof=motors_dof, qpos=qpos[:-2], display_video=display_video)
    
    # Grasp
    execute_steps(franka, scene, cam, end_effector, cube, logger, 
                 num_steps=100, motors_dof=motors_dof, qpos=qpos[:-2],
                 finger_force=np.array([-0.5, -0.5]), fingers_dof=fingers_dof,
                 print_status=True, print_interval=20, phase_name="Grasping", display_video=display_video)
    
    # Lift
    qpos = franka.inverse_kinematics(
        link=end_effector,
        pos=np.array([0.65, 0.0, 0.28]),
        quat=np.array([0, 1, 0, 0]),
    )
    execute_steps(franka, scene, cam, end_effector, cube, logger, 
                 num_steps=200, motors_dof=motors_dof, qpos=qpos[:-2],
                 print_status=True, print_interval=40, phase_name="Lifting", display_video=display_video)

    # Plan and execute a multi-waypoint transfer trajectory while holding the cube
    transfer_waypoints = [
        {"pos": [0.65, 0.0, 0.32], "quat": [0, 1, 0, 0], "steps": 120},
        {"pos": [0.55, 0.25, 0.32], "quat": [0, 1, 0, 0], "steps": 180},
        {"pos": [0.55, 0.25, 0.18], "quat": [0, 1, 0, 0], "steps": 140},
        {"pos": [0.55, 0.25, 0.15], "quat": [0, 1, 0, 0], "steps": 80},
    ]

    path = generate_composite_trajectory(
        franka,
        end_effector,
        transfer_waypoints,
        default_steps=120,
        finger_qpos=0.0,  # keep gripper closed during transport
    )
    execute_trajectory(franka, scene, cam, end_effector, cube, logger, path, display_video=display_video)

    final_arm_qpos = path[-1][:-2]

    # Release cube at the final waypoint and stabilize
    execute_steps(
        franka,
        scene,
        cam,
        end_effector,
        cube,
        logger,
        num_steps=120,
        motors_dof=motors_dof,
        qpos=final_arm_qpos,
        finger_force=np.array([0.6, 0.6]),
        fingers_dof=fingers_dof,
        print_status=True,
        print_interval=40,
        phase_name="Releasing",
        display_video=display_video,
    )

    # Retreat upwards to a safe pose after placement
    retreat_qpos = franka.inverse_kinematics(
        link=end_effector,
        pos=np.array([0.55, 0.25, 0.30]),
        quat=np.array([0, 1, 0, 0]),
    )
    retreat_qpos[-2:] = 0.04
    execute_steps(
        franka,
        scene,
        cam,
        end_effector,
        cube,
        logger,
        num_steps=120,
        motors_dof=motors_dof,
        qpos=retreat_qpos[:-2],
        print_status=False,
        display_video=display_video,
        phase_name="Retreat",
    )
