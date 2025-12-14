"""Pick-and-place demonstration functionality."""
import numpy as np
from .trajectory import execute_trajectory, generate_composite_trajectory
from .steps import execute_steps
from src.utils import (
    print_ik_target_vs_current,
    draw_spawn_area,
    draw_drop_area,
    draw_trajectory_debug,
    erase_trajectory_debug,
    log_transfer_debug,
)


def _as_np(vec):
    if hasattr(vec, "detach") and hasattr(vec, "cpu"):
        return vec.detach().cpu().numpy()
    return np.asarray(vec, dtype=float)


def _get_gripper_finger_tip_offset():
    """
    Get the Z-offset from hand link center to gripper finger tips (when closed).
    
    Franka gripper specs:
    - Finger length: ~0.1m from hand mounting point
    - When fully open: ~0.04m apart (0.02m per finger)
    - Hand center to closed finger tip: ~0.105m downward
    
    Returns the Z offset (negative, downward from hand link center).
    """
    return -0.085# Approximate distance from hand link to closed finger tip


# Removed mass manipulation helpers: feature deprecated


def _slerp_quat(q0, q1, t):
    """Spherical linear interpolation between two unit quaternions."""
    q0 = np.asarray(q0, dtype=float)
    q1 = np.asarray(q1, dtype=float)
    dot = np.dot(q0, q1)
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    dot = np.clip(dot, -1.0, 1.0)
    theta = np.arccos(dot)
    if np.abs(theta) < 1e-6:
        return q0 + t * (q1 - q0)
    sin_theta = np.sin(theta)
    return (np.sin((1.0 - t) * theta) / sin_theta) * q0 + (np.sin(t * theta) / sin_theta) * q1


def _quat_to_rotation_matrix(quat):
    """
    Convert quaternion to 3x3 rotation matrix.
    
    Args:
        quat: Quaternion in (w, x, y, z) format
    
    Returns:
        3x3 rotation matrix
    """
    q = np.asarray(quat, dtype=float)
    w, x, y, z = q
    
    R = np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
        [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)]
    ])
    
    return R


def _rotation_matrix_to_quat(R):
    """
    Convert 3x3 rotation matrix to quaternion.
    
    Args:
        R: 3x3 rotation matrix
    
    Returns:
        Quaternion in (w, x, y, z) format
    """
    trace = np.trace(R)
    
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    
    quat = np.array([w, x, y, z])
    return quat / np.linalg.norm(quat)


def _align_gripper_to_cube_z_axis(cube_quat):
    """
    Create gripper orientation that is face-down but rotated around world z-axis
    to align with the cube's orientation (projected onto horizontal plane).
    
    The gripper remains looking down, but rotates around the vertical (world z) axis
    so the fingers align with the cube's sides.
    
    Args:
        cube_quat: Cube's quaternion in (w, x, y, z) format
    
    Returns:
        Gripper quaternion in (w, x, y, z) format
    """
    # Get cube's rotation matrix
    R_cube = _quat_to_rotation_matrix(cube_quat)
    
    # Extract cube's x-axis in world coordinates (1st column of rotation matrix)
    # Project it onto the horizontal (xy) plane
    cube_x_world = R_cube[:, 0]
    cube_x_horizontal = np.array([cube_x_world[0], cube_x_world[1], 0.0])
    
    # If the projection is too small, use cube's y-axis instead
    if np.linalg.norm(cube_x_horizontal) < 0.1:
        cube_y_world = R_cube[:, 1]
        cube_x_horizontal = np.array([cube_y_world[0], cube_y_world[1], 0.0])
    
    # If still too small (cube is perfectly aligned vertically), use world x-axis
    if np.linalg.norm(cube_x_horizontal) < 0.1:
        cube_x_horizontal = np.array([1.0, 0.0, 0.0])
    
    # Normalize to get the rotation angle around world z-axis
    cube_x_horizontal = cube_x_horizontal / np.linalg.norm(cube_x_horizontal)
    
    # Calculate rotation angle around world z-axis
    # Reference direction is world +x axis
    angle = np.arctan2(cube_x_horizontal[1], cube_x_horizontal[0])
    
    # Create face-down orientation with z-rotation
    # Face-down: rotate 180° around y-axis, then rotate around z-axis
    # This is: R_z(angle) * R_y(180°)
    
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    
    # R_y(180°) = rotation matrix for 180° around y-axis
    # Then apply R_z(angle)
    # Combined rotation matrix:
    R_gripper = np.array([
        [-cos_a, -sin_a, 0.0],
        [-sin_a,  cos_a, 0.0],
        [   0.0,    0.0, -1.0]
    ])
    
    return _rotation_matrix_to_quat(R_gripper)


def _mirror_gripper_orientation(gripper_quat, angle_deg=90.0):
    """
    Create a rotated gripper orientation (rotation around world z-axis).
    
    Args:
        gripper_quat: Original gripper quaternion in (w, x, y, z) format
        angle_deg: Rotation angle in degrees (default 90° instead of 180° to reduce IK jumps)
    
    Returns:
        Rotated quaternion in (w, x, y, z) format
    """
    R = _quat_to_rotation_matrix(gripper_quat)
    
    # Rotate around world z-axis by the specified angle
    angle_rad = np.radians(angle_deg)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    
    # Rotation matrix around z-axis
    R_z = np.array([
        [cos_a, -sin_a, 0.0],
        [sin_a,  cos_a, 0.0],
        [  0.0,    0.0, 1.0]
    ])
    
    # Apply rotation: R_new = R_z * R_original
    R_rotated = R_z @ R
    
    return _rotation_matrix_to_quat(R_rotated)


def _generate_arc_transfer_waypoints(
    start_pos,
    end_pos,
    quat,
    quat_end=None,
    arc_height=0.55,
    num_points=24,
    target_total_steps=240,
):
    """
    Generate a smooth arc tilted at ~45° away from robot base, with Slerp'd orientation.

    - Arc plane is tilted away from base at ~45° ± random 10°, not vertical.
    - Control point creates a parabolic bulge in that plane.
    - Orientation interpolated via Slerp from quat_start to quat_end.
    """
    p0 = np.asarray(start_pos, dtype=float)
    p1 = np.asarray(end_pos, dtype=float)
    quat_start = np.asarray(quat, dtype=float)
    quat_end_arr = np.asarray(quat_end, dtype=float) if quat_end is not None else quat_start.copy()

    radial_xy = np.array([p0[0] + p1[0], p0[1] + p1[1]], dtype=float)
    radial_norm = np.linalg.norm(radial_xy)
    if radial_norm > 1e-4:
        radial_dir = radial_xy / radial_norm
    else:
        radial_dir = np.array([1.0, 0.0], dtype=float)

    up = np.array([0.0, 0.0, 1.0], dtype=float)
    # Randomize tilt angle ±10 degrees around 45°
    tilt_angle = np.radians(45.0 + np.random.uniform(-10.0, 10.0))
    tilt_dir_xy = radial_dir
    arc_normal = np.array([
        tilt_dir_xy[0] * np.cos(tilt_angle),
        tilt_dir_xy[1] * np.cos(tilt_angle),
        np.sin(tilt_angle)
    ], dtype=float)

    ctrl = 0.5 * (p0 + p1)
    ctrl = ctrl + arc_height * arc_normal

    ts = np.linspace(0.0, 1.0, num_points)
    steps_per = int(np.clip(target_total_steps / max(num_points - 1, 1), 4, 40))

    arc_waypoints = []
    for t in ts[1:]:  # skip p0 (already current)
        pos = (1 - t) * (1 - t) * p0 + 2 * (1 - t) * t * ctrl + t * t * p1
        quat_interp = _slerp_quat(quat_start, quat_end_arr, t)
        arc_waypoints.append({"pos": pos.tolist(), "quat": quat_interp.tolist(), "steps": steps_per})

    if arc_waypoints:
        arc_waypoints[-1]["pos"] = p1.tolist()

    return arc_waypoints


def run_pick_and_place_demo(
    franka,
    scene,
    cam,
    end_effector,
    cube,
    logger,
    motors_dof,
    fingers_dof,
    display_video=True,
    drop_pos=None,
    debug_plot_transfer=True,
):
    """
    Execute pick-and-place for a single cube to a specified drop position.
    """
    cube_pos = _as_np(cube.get_pos())
    cube_quat = _as_np(cube.get_quat())
    drop_pos = _as_np(drop_pos) if drop_pos is not None else np.array([0.55, 0.25, 0.15])
    
    # Align gripper to match cube's z-axis orientation in world coordinates
    quat_grasp = _align_gripper_to_cube_z_axis(cube_quat)
    
    # Create rotated orientation for drop-off (45° rotation instead of 90° to reduce IK jumps)
    quat_drop = _mirror_gripper_orientation(quat_grasp, angle_deg=45.0)
    
    # Get cube's orientation for debugging
    R_cube = _quat_to_rotation_matrix(cube_quat)
    cube_x_world = R_cube[:, 0]
    angle_grasp = np.arctan2(cube_x_world[1], cube_x_world[0])
    print(f"[DEBUG] Cube x-axis rotation (world z): {np.degrees(angle_grasp):.2f}°")
    print(f"[DEBUG] Grasp quaternion: [{quat_grasp[0]:.3f}, {quat_grasp[1]:.3f}, {quat_grasp[2]:.3f}, {quat_grasp[3]:.3f}]")
    print(f"[DEBUG] Drop quaternion: [{quat_drop[0]:.3f}, {quat_drop[1]:.3f}, {quat_drop[2]:.3f}, {quat_drop[3]:.3f}]")

    # Cube dimensions: 0.04m x 0.04m x 0.04m
    # cube_pos is the center, so half-height = 0.02m
    cube_half_height = 0.025
    cube_top_z = cube_pos[2] + cube_half_height
    
    # Gripper finger tip offset from hand link center
    # hand link is above the actual closing point of fingers
    finger_tip_offset = _get_gripper_finger_tip_offset()  # -0.105m (negative = downward)

    lateral_offset = np.random.uniform(-0.015, 0.015, size=2)
    # height_jitter = np.random.uniform(-0.01, 0.015)

    # Heights for hand link center (IK targets the hand link, not finger tips)
    # finger_tip_offset is -0.105 (downward from hand link center)
    # To position fingers at height H: hand_link_z = H - finger_tip_offset = H + 0.105
    hover_height = cube_top_z + 0.20 + abs(finger_tip_offset)  # Fingers 0.20m above cube
    approach_height = cube_top_z + abs(finger_tip_offset) + np.random.uniform(0.005, 0.01)  
    lift_height = cube_top_z + 0.15 + abs(finger_tip_offset)  # Fingers 0.10m above cube

    # Move to hover above the cube
    hover_target_pos = np.array([cube_pos[0], cube_pos[1], hover_height])
    print(f"[DEBUG] Cube top Z: {cube_top_z:.4f}, Hand link hover Z: {hover_height:.4f}, Finger tip Z: {hover_height + finger_tip_offset:.4f}")
    q_hover = franka.inverse_kinematics(
        link=end_effector,
        pos=hover_target_pos,
        quat=quat_grasp,
    )
    print(f"[DEBUG] Hover IK result: {q_hover}")
    q_hover[-2:] = 0.04
    print(f"[DEBUG] Hover IK with gripper: {q_hover}")
    path = franka.plan_path(qpos_goal=q_hover, num_waypoints=300)  # Increased from 200 for smoother motion with damped controller
    print(f"[DEBUG] Hover trajectory has {len(path)} waypoints")
    execute_trajectory(franka, scene, cam, end_effector, cube, logger, path, display_video=display_video, phase_name="Hover")

    # Stabilize at hover
    execute_steps(franka, scene, cam, end_effector, cube, logger, num_steps=80, display_video=display_video, phase_name="Hover Stabilize")

    # Descend toward the cube
    approach_target_pos = np.array([
        cube_pos[0] + lateral_offset[0],
        cube_pos[1] + lateral_offset[1],
        approach_height,
    ])
    print(f"[DEBUG] Approach target: pos={approach_target_pos}, lateral_offset={lateral_offset}")
    print(f"[DEBUG] Approach: Hand link Z: {approach_height:.4f}, Finger tip Z: {approach_height + finger_tip_offset:.4f}, Cube top Z: {cube_top_z:.4f}")
    q_approach = franka.inverse_kinematics(
        link=end_effector,
        pos=approach_target_pos,
        quat=quat_grasp,
    )
    print(f"[DEBUG] Approach IK result: {q_approach}")
    q_current = franka.get_qpos()
    if hasattr(q_current, 'cpu'):
        q_current = q_current.cpu().numpy()
    print_ik_target_vs_current(franka, q_approach[:-2], q_current, "Approach")
    execute_steps(
        franka,
        scene,
        cam,
        end_effector,
        cube,
        logger,
        num_steps=150,
        motors_dof=motors_dof,
        qpos=q_approach[:-2],
        display_video=display_video,
        debug=True,
        phase_name="Approach",
    )

    # Grasp
    logger.mark_phase_start("Grasping")
    execute_steps(
        franka,
        scene,
        cam,
        end_effector,
        cube,
        logger,
        num_steps=150,
        motors_dof=motors_dof,
        qpos=q_approach[:-2],
        finger_force=np.array([-0.5, -0.5]),
        fingers_dof=fingers_dof,
        print_status=True,
        print_interval=30,
        phase_name="Grasping",
        display_video=display_video,
    )
    logger.mark_phase_end("Grasping")

    # Lift straight up
    q_lift = franka.inverse_kinematics(
        link=end_effector,
        pos=np.array([cube_pos[0], cube_pos[1], lift_height]),
        quat=quat_grasp,
    )
    execute_steps(
        franka,
        scene,
        cam,
        end_effector,
        cube,
        logger,
        num_steps=200,
        motors_dof=motors_dof,
        qpos=q_lift[:-2],
        print_status=True,
        print_interval=50,
        phase_name="Lifting",
        display_video=display_video,
    )

    # Plan and execute transport to drop site with a robust arc
    hover_drop_z = drop_pos[2] + 0.05  # End arc just 5cm above drop position
    start_pos = np.array([cube_pos[0], cube_pos[1], lift_height])
    end_hover_pos = np.array([drop_pos[0], drop_pos[1], hover_drop_z])

    transfer_waypoints = _generate_arc_transfer_waypoints(
        start_pos=start_pos,
        end_pos=end_hover_pos,
        quat=quat_grasp,
        quat_end=quat_drop,
        arc_height=np.random.uniform(0.55, 0.65),
        num_points=24,
        target_total_steps=24,
    )

    # Final descent to place the cube
    transfer_waypoints.append({"pos": [drop_pos[0], drop_pos[1], drop_pos[2]], "quat": quat_drop, "steps": 90})

    # Draw trajectory debug visualization
    debug_trajectory_lines = draw_trajectory_debug(scene, transfer_waypoints, color=(1, 1, 0, 1))

    # Get current joint state to ensure trajectory continuity
    q_current_before_transfer = _as_np(franka.get_qpos())
    
    path = generate_composite_trajectory(
        franka,
        end_effector,
        transfer_waypoints,
        default_steps=150,
        finger_qpos=0.0,
    )

    # Check for large IK jump at trajectory start and insert smooth transition if needed
    if len(path) > 0:
        first_q = path[0]
        joint_delta = np.linalg.norm(q_current_before_transfer[:7] - first_q[:7])
        if joint_delta > 0.5:  # Large discontinuity detected
            print(f"[DEBUG] Large IK jump at transport start (delta={joint_delta:.3f} rad). Inserting smooth transition.")
            n_transition = 15
            transition = np.linspace(q_current_before_transfer, first_q, n_transition + 1)[1:]
            path = np.vstack([transition, path])

    if debug_plot_transfer:
        log_transfer_debug(transfer_waypoints, path, franka, end_effector)

    logger.mark_phase_start("Transport")
    execute_trajectory(
        franka,
        scene,
        cam,
        end_effector,
        cube,
        logger,
        path,
        display_video=display_video,
        step_callbacks=None,
        phase_name="Transport",
    )
    logger.mark_phase_end("Transport")

    final_arm_qpos = path[-1][:-2]

    # Release cube at the final waypoint and stabilize
    logger.mark_phase_start("Releasing")
    execute_steps(
        franka,
        scene,
        cam,
        end_effector,
        cube,
        logger,
        num_steps=150,
        motors_dof=motors_dof,
        qpos=final_arm_qpos,
        finger_force=np.array([0.6, 0.6]),
        fingers_dof=fingers_dof,
        print_status=True,
        print_interval=40,
        phase_name="Releasing",
        display_video=display_video,
    )
    logger.mark_phase_end("Releasing")
    
    # Display force graph for this pick-and-place cycle
    print("[DEMO] Cube dropped. Displaying force telemetry...")
    releasing_viz_key = f"Releasing_cycle{logger.cycle_count}"
    if releasing_viz_key in logger.phase_visualizers:
        viz = logger.phase_visualizers[releasing_viz_key]
        print(f"[DEMO] Force data collected: {len(viz.timestamps)} timesteps")
        if len(viz.timestamps) > 0:
            print(f"[DEMO] Left force range: {min(viz.left_forces):.3f} - {max(viz.left_forces):.3f} N")
            print(f"[DEMO] Right force range: {min(viz.right_forces):.3f} - {max(viz.right_forces):.3f} N")
            viz.plot(block=True)
        else:
            print("[DEMO] No force data to display!")
    else:
        print(f"[DEMO] No visualizer found for key: {releasing_viz_key}")
    
    print("[DEMO] Resuming with next cube...")
    logger.cycle_count += 1

    # Erase trajectory debug visualization after cube is dropped
    erase_trajectory_debug(scene, debug_trajectory_lines)

    # Retreat upwards to a safe pose after placement
    retreat_qpos = franka.inverse_kinematics(
        link=end_effector,
        pos=np.array([drop_pos[0], drop_pos[1], drop_pos[2] + 0.16]),
        quat=quat_drop,
    )
    retreat_qpos[-2:] = 0.04
    execute_steps(
        franka,
        scene,
        cam,
        end_effector,
        cube,
        logger,
        num_steps=130,
        motors_dof=motors_dof,
        qpos=retreat_qpos[:-2],
        display_video=display_video,
        phase_name="Retreat",
    )


def run_iterative_pick_and_place(
    franka,
    scene,
    cam,
    end_effector,
    cubes,
    logger,
    motors_dof,
    fingers_dof,
    display_video=True,
):
    """Iteratively pick random cubes and place them to the robot's side."""
    if not cubes:
        print("No cubes to manipulate.")
        return

    # Draw spawn and drop areas
    # _ = draw_spawn_area(scene, x_range=(0.35, 0.55), y_range=(-0.55, -0.05), z_height=0.15)
    
    drop_base = np.array([0.55, 0.38, 0.14])
    drop_step = np.array([0.0, 0.0, 0.0])
    
    # Draw drop area for the first drop position
    # _ = draw_drop_area(scene, drop_pos=drop_base, area_size=0.05, z_height=0.20)

    remaining = list(cubes)
    for i in range(len(remaining)):
        cube_idx = np.random.randint(0, len(remaining))
        cube = remaining.pop(cube_idx)
        drop_pos = drop_base + i * drop_step
        print(f"Handling cube {i+1}/{len(cubes)} -> drop at {drop_pos}")

        run_pick_and_place_demo(
            franka,
            scene,
            cam,
            end_effector,
            cube,
            logger,
            motors_dof,
            fingers_dof,
            display_video=display_video,
            drop_pos=drop_pos,
            debug_plot_transfer=True,
        )

        # Despawn placed cube
        try:
            scene.remove_entity(cube)
        except Exception:
            cube.set_pos(np.array([2.0 + i, 2.0, 0.05]))
        print(f"Cube {i+1} placed and removed.")
