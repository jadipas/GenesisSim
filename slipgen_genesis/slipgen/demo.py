"""Pick-and-place demonstration functionality."""
import numpy as np
from slipgen.trajectory import execute_trajectory, generate_composite_trajectory, execute
from slipgen.steps import execute_steps
from slipgen.debug_utils import (
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
    return -0.085


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
    """Convert quaternion to 3x3 rotation matrix."""
    q = np.asarray(quat, dtype=float)
    w, x, y, z = q
    
    R = np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
        [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)]
    ])
    
    return R


def _rotation_matrix_to_quat(R):
    """Convert 3x3 rotation matrix to quaternion."""
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
    """Create gripper orientation aligned with cube's z-axis."""
    R_cube = _quat_to_rotation_matrix(cube_quat)
    
    cube_x_world = R_cube[:, 0]
    cube_x_horizontal = np.array([cube_x_world[0], cube_x_world[1], 0.0])
    
    if np.linalg.norm(cube_x_horizontal) < 0.1:
        cube_y_world = R_cube[:, 1]
        cube_x_horizontal = np.array([cube_y_world[0], cube_y_world[1], 0.0])
    
    if np.linalg.norm(cube_x_horizontal) < 0.1:
        cube_x_horizontal = np.array([1.0, 0.0, 0.0])
    
    cube_x_horizontal = cube_x_horizontal / np.linalg.norm(cube_x_horizontal)
    
    angle = np.arctan2(cube_x_horizontal[1], cube_x_horizontal[0])
    
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    
    R_gripper = np.array([
        [-cos_a, -sin_a, 0.0],
        [-sin_a,  cos_a, 0.0],
        [   0.0,    0.0, -1.0]
    ])
    
    return _rotation_matrix_to_quat(R_gripper)


def _mirror_gripper_orientation(gripper_quat, angle_deg=90.0):
    """Create a rotated gripper orientation (rotation around world z-axis)."""
    R = _quat_to_rotation_matrix(gripper_quat)
    
    angle_rad = np.radians(angle_deg)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    
    R_z = np.array([
        [cos_a, -sin_a, 0.0],
        [sin_a,  cos_a, 0.0],
        [  0.0,    0.0, 1.0]
    ])
    
    R_rotated = R_z @ R
    
    return _rotation_matrix_to_quat(R_rotated)


def apply_phase_warp_bump(path: np.ndarray, level: int) -> np.ndarray:
    """
    Keep same number of steps, but traverse mid-section faster via a smooth phase warp.
    level: 0=no bump, 1 mild, 2 medium, 3 strong
    """
    path = np.asarray(path)
    n = len(path)
    if level <= 0 or n < 20:
        return path

    # bump strength: larger -> stronger speedup in the middle
    a = {1: 0.35, 2: 0.65, 3: 1.0}.get(level, 0.35)

    # normalized time grid
    t = np.linspace(0.0, 1.0, n)

    # Smooth "S" bump centered at 0.5: derivative increases near the center
    # We implement a phase warp using a tanh-based cumulative mapping.
    k = 6.0  # sharpness of the bump
    bump = np.tanh(k * (t - 0.5))
    bump = (bump - bump.min()) / (bump.max() - bump.min())  # [0,1]
    # Combine with identity; 'a' controls how much we deviate from uniform phase
    phi = (1 - a) * t + a * bump

    # Ensure strict monotonicity and endpoints
    phi[0], phi[-1] = 0.0, 1.0
    phi = np.maximum.accumulate(phi)

    # Sample original path at indices corresponding to phi
    idx_f = phi * (n - 1)
    idx0 = np.floor(idx_f).astype(int)
    idx1 = np.clip(idx0 + 1, 0, n - 1)
    w = idx_f - idx0

    # Linear interpolation in joint space (arm joints only)
    # Preserve gripper DOFs to avoid random opening during transport
    if path.shape[1] > 7:  # Has gripper DOFs
        arm_warped = (1 - w)[:, None] * path[idx0, :7] + w[:, None] * path[idx1, :7]
        # Take gripper values from original path (no interpolation)
        gripper_original = path[:, 7:]
        warped = np.hstack([arm_warped, gripper_original])
    else:
        warped = (1 - w)[:, None] * path[idx0] + w[:, None] * path[idx1]
    return warped


def add_joint_shake(q: np.ndarray, step_idx: int, total_steps: int,
                    amp: float = 0.015, freq: float = 6.0,
                    joints=(3, 5)) -> np.ndarray:
    """
    Small sinusoidal shake on selected arm joints.
    amp in radians. freq in cycles over the full trajectory.
    Only affects arm joints (0-6), never gripper DOFs (7-8).
    """
    q2 = q.copy()
    phase = 2 * np.pi * freq * (step_idx / max(1, total_steps - 1))
    s = np.sin(phase)
    for j in joints:
        if j < 7:  # Only apply to arm joints, never gripper
            q2[j] += amp * s
    return q2


def _generate_arc_transfer_waypoints(
    start_pos,
    end_pos,
    quat,
    quat_end=None,
    arc_height=0.55,
    num_points=24,
    target_total_steps=240,
    arc_bias=0.0,
):
    """Generate a smooth arc tilted at ~45° away from robot base with optional lateral bias."""
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
    tilt_angle = np.radians(45.0 + np.random.uniform(-10.0, 10.0))
    tilt_dir_xy = radial_dir
    arc_normal = np.array([
        tilt_dir_xy[0] * np.cos(tilt_angle),
        tilt_dir_xy[1] * np.cos(tilt_angle),
        np.sin(tilt_angle)
    ], dtype=float)

    ctrl = 0.5 * (p0 + p1)
    ctrl = ctrl + arc_height * arc_normal
    
    # Add lateral bias for tangential stress
    if abs(arc_bias) > 1e-6:
        lateral_dir = np.array([-radial_dir[1], radial_dir[0], 0.0], dtype=float)
        ctrl = ctrl + arc_bias * lateral_dir

    ts = np.linspace(0.0, 1.0, num_points)
    steps_per = int(np.clip(target_total_steps / max(num_points - 1, 1), 4, 40))

    arc_waypoints = []
    for t in ts[1:]:
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
    transport_steps=240,
    phase_warp_level=0,
    shake_amp=0.0,
    shake_freq=6.0,
    knobs=None,
):
    """
    Execute pick-and-place for a single cube to a specified drop position.
    
    Transport disturbance parameters:
    - transport_steps: Total steps for transport phase (default 240)
    - phase_warp_level: 0=none, 1=mild, 2=medium, 3=strong speed-up in middle
    - shake_amp: Joint shake amplitude in radians (0=off, 0.01-0.03 typical)
    - shake_freq: Shake frequency in cycles over trajectory (4-8 typical)
    """
    # Reset visualizer for this pick-and-place cycle
    logger.reset_visualizer()
    
    cube_pos = _as_np(cube.get_pos())
    cube_quat = _as_np(cube.get_quat())
    drop_pos = _as_np(drop_pos) if drop_pos is not None else np.array([0.55, 0.25, 0.15])
    
    quat_grasp = _align_gripper_to_cube_z_axis(cube_quat)
    quat_drop = _mirror_gripper_orientation(quat_grasp, angle_deg=45.0)
    
    R_cube = _quat_to_rotation_matrix(cube_quat)
    cube_x_world = R_cube[:, 0]
    angle_grasp = np.arctan2(cube_x_world[1], cube_x_world[0])
    print(f"[DEBUG] Cube x-axis rotation (world z): {np.degrees(angle_grasp):.2f}°")
    print(f"[DEBUG] Grasp quaternion: [{quat_grasp[0]:.3f}, {quat_grasp[1]:.3f}, {quat_grasp[2]:.3f}, {quat_grasp[3]:.3f}]")
    print(f"[DEBUG] Drop quaternion: [{quat_drop[0]:.3f}, {quat_drop[1]:.3f}, {quat_drop[2]:.3f}, {quat_drop[3]:.3f}]")

    cube_half_height = 0.025
    cube_top_z = cube_pos[2] + cube_half_height
    
    finger_tip_offset = _get_gripper_finger_tip_offset()

    lateral_offset = np.random.uniform(-0.015, 0.015, size=2)

    hover_height = cube_top_z + 0.20 + abs(finger_tip_offset)
    approach_height = cube_top_z + abs(finger_tip_offset) + np.random.uniform(0.005, 0.01)  
    lift_height = cube_top_z + 0.15 + abs(finger_tip_offset)

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
    path = franka.plan_path(qpos_goal=q_hover, num_waypoints=300)
    print(f"[DEBUG] Hover trajectory has {len(path)} waypoints")
    execute_trajectory(franka, scene, cam, end_effector, cube, logger, path, display_video=display_video, phase_name="Hover")

    execute_steps(franka, scene, cam, end_effector, cube, logger, num_steps=80, display_video=display_video, phase_name="Hover Stabilize", knobs=knobs)

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
        knobs=knobs,
    )

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
        finger_force=np.array([-15.0, -15.0]),
        fingers_dof=fingers_dof,
        print_status=True,
        print_interval=30,
        phase_name="Grasping",
        display_video=display_video,
        knobs=knobs,
    )
    logger.mark_phase_end("Grasping")

    # Stabilize gripper: hold arm in place and let gripper force settle before transport
    logger.mark_phase_start("Grasp Stabilization")
    q_current = franka.get_qpos()
    if hasattr(q_current, 'cpu'):
        q_current = q_current.cpu().numpy()
    execute_steps(
        franka,
        scene,
        cam,
        end_effector,
        cube,
        logger,
        num_steps=100,
        motors_dof=motors_dof,
        qpos=q_current[:-2],  
        finger_force=np.array([-15.0, -15.0]),  
        fingers_dof=fingers_dof,
        print_status=True,
        print_interval=25,
        phase_name="Grasp Stabilization",
        display_video=display_video,
        knobs=knobs,
    )
    logger.mark_phase_end("Grasp Stabilization")

    # Capture final gripper position for transport (position control)
    q_stable = franka.get_qpos()
    if hasattr(q_stable, 'cpu'):
        q_stable = q_stable.cpu().numpy()
    finger_qpos_for_transport = q_stable[-2:]  # Lock fingers at this position during transport

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
        knobs=knobs,
    )

    hover_drop_z = drop_pos[2] + 0.05
    start_pos = np.array([cube_pos[0], cube_pos[1], lift_height])
    end_hover_pos = np.array([drop_pos[0], drop_pos[1], hover_drop_z])

    transfer_waypoints = _generate_arc_transfer_waypoints(
        start_pos=start_pos,
        end_pos=end_hover_pos,
        quat=quat_grasp,
        quat_end=quat_drop,
        arc_height=np.random.uniform(0.55, 0.65),
        num_points=24,
        target_total_steps=transport_steps,  # Now configurable (default 240)
        arc_bias=np.random.uniform(-0.03, 0.03),  # Lateral curvature for tangential stress
    )

    transfer_waypoints.append({"pos": [drop_pos[0], drop_pos[1], drop_pos[2]], "quat": quat_drop, "steps": 90})

    debug_trajectory_lines = draw_trajectory_debug(scene, transfer_waypoints, color=(1, 1, 0, 1))

    q_current_before_transfer = _as_np(franka.get_qpos())
    
    # Generate trajectory with gripper position-locked at stabilized position
    path = generate_composite_trajectory(
        franka,
        end_effector,
        transfer_waypoints,
        default_steps=150,
        finger_qpos=finger_qpos_for_transport,  # Lock gripper at stabilized position during transport
    )

    if len(path) > 0:
        first_q = path[0]
        joint_delta = np.linalg.norm(q_current_before_transfer[:7] - first_q[:7])
        if joint_delta > 0.5:
            print(f"[DEBUG] Large IK jump at transport start (delta={joint_delta:.3f} rad). Inserting smooth transition.")
            n_transition = 15
            transition = np.linspace(q_current_before_transfer, first_q, n_transition + 1)[1:]
            path = np.vstack([transition, path])

    if debug_plot_transfer:
        log_transfer_debug(transfer_waypoints, path, franka, end_effector)

    logger.mark_phase_start("Transport")
    # Use execute() with disturbance parameters for repeatable slip stress
    # Gripper now uses position control (locked at stabilized position)
    # The path already contains the locked finger positions from generate_composite_trajectory
    execute(
        franka,
        scene,
        cam,
        end_effector,
        cube,
        logger,
        path,
        display_video=display_video,
        check_contact=True,
        step_callbacks=None,
        phase_name="Transport",
        debug=False,
        disturb_level=phase_warp_level,  # Apply phase warp for speed variation
        shake_amp=shake_amp,              # Apply joint shake for inertial disturbance
        shake_freq=shake_freq,
        knobs=knobs,
        finger_force=np.array([-15.0, -15.0]),  
        fingers_dof=fingers_dof,
    )
    logger.mark_phase_end("Transport")

    final_arm_qpos = path[-1][:-2]

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
        finger_force=np.array([6.0, 6.0]),
        fingers_dof=fingers_dof,
        print_status=True,
        print_interval=40,
        phase_name="Releasing",
        display_video=display_video,
        knobs=knobs,
    )
    logger.mark_phase_end("Releasing")
    
    print("[DEMO] Cube dropped. Generating force plot...")
    
    # Generate force plot for this pick-and-place cycle
    filename = f"force_plot_cycle{logger.cycle_count}.png"
    logger.save_force_plot(output_dir=".", filename=filename)
    
    logger.cycle_count += 1

    erase_trajectory_debug(scene, debug_trajectory_lines)

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
        knobs=knobs,
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
    transport_steps=240,
    phase_warp_level=0,
    shake_amp=0.0,
    shake_freq=6.0,
    knobs=None,
):
    """
    Iteratively pick random cubes and place them to the robot's side.
    
    Transport disturbance parameters:
    - transport_steps: Total steps for transport phase (default 240)
    - phase_warp_level: 0=none, 1=mild, 2=medium, 3=strong speed-up in middle
    - shake_amp: Joint shake amplitude in radians (0=off, 0.01-0.03 typical)
    - shake_freq: Shake frequency in cycles over trajectory (4-8 typical)
    """
    if not cubes:
        print("No cubes to manipulate.")
        return

    drop_base = np.array([0.55, 0.38, 0.14])
    drop_step = np.array([0.0, 0.0, 0.0])
    
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
            transport_steps=transport_steps,
            phase_warp_level=phase_warp_level,
            shake_amp=shake_amp,
            shake_freq=shake_freq,
            knobs=knobs,
        )

        try:
            scene.remove_entity(cube)
        except Exception:
            cube.set_pos(np.array([2.0 + i, 2.0, 0.05]))
        print(f"Cube {i+1} placed and removed.")
