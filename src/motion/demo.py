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
    return -0.105  # Approximate distance from hand link to closed finger tip


def _get_mass(cube):
    for getter in ("get_mass", "mass", "get_mass_properties"):
        try:
            val = getattr(cube, getter)
            if callable(val):
                out = val()
            else:
                out = val
            # Some APIs return tuple (mass, inertia)
            if isinstance(out, (list, tuple)):
                return float(out[0])
            return float(out)
        except Exception:
            continue
    return None


def _set_mass(cube, value):
    for setter in ("set_mass", "set_property"):
        try:
            fn = getattr(cube, setter)
            if setter == "set_property":
                fn("mass", value)
            else:
                fn(value)
            return True
        except Exception:
            continue
    try:
        cube.mass = value
        return True
    except Exception:
        return False


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


def _generate_arc_transfer_waypoints(
    start_pos,
    end_pos,
    quat,
    quat_end=None,
    arc_height=0.18,
    num_points=24,
    target_total_steps=240,
):
    """
    Generate a smooth arc tilted at ~45° away from robot base, with Slerp'd orientation.

    - Arc plane is tilted away from base at ~45°, not vertical.
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
    tilt_angle = np.radians(45.0)
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
    drop_pos = _as_np(drop_pos) if drop_pos is not None else np.array([0.55, 0.25, 0.15])
    quat_ny = np.array([0, 1, 0, 0])  # fixed face-down orientation

    # Cube dimensions: 0.04m x 0.04m x 0.04m
    # cube_pos is the center, so half-height = 0.02m
    cube_half_height = 0.02
    cube_top_z = cube_pos[2] + cube_half_height
    
    # Gripper finger tip offset from hand link center
    # hand link is above the actual closing point of fingers
    finger_tip_offset = _get_gripper_finger_tip_offset()

    lateral_offset = np.random.uniform(-0.005, 0.005, size=2)
    # height_jitter = np.random.uniform(-0.01, 0.015)

    # Heights for hand link center (IK targets the hand link, not finger tips)
    # We need to add finger_tip_offset to get proper clearance
    hover_height = cube_top_z + 0.20 - finger_tip_offset
    approach_height = cube_top_z + 0.005 - finger_tip_offset  # Fingers just above cube top
    lift_height = cube_top_z + 0.26 - finger_tip_offset

    # Move to hover above the cube
    hover_target_pos = np.array([cube_pos[0], cube_pos[1], hover_height])
    print(f"[DEBUG] Cube top Z: {cube_top_z:.4f}, Hand link hover Z: {hover_height:.4f}, Finger tip Z: {hover_height + finger_tip_offset:.4f}")
    q_hover = franka.inverse_kinematics(
        link=end_effector,
        pos=hover_target_pos,
        quat=quat_ny,
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
        quat=quat_ny,
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
        num_steps=90,
        motors_dof=motors_dof,
        qpos=q_approach[:-2],
        display_video=display_video,
        debug=True,
        phase_name="Approach",
    )

    # Grasp
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

    # Lift straight up
    q_lift = franka.inverse_kinematics(
        link=end_effector,
        pos=np.array([cube_pos[0], cube_pos[1], lift_height]),
        quat=quat_ny,
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
    hover_drop_z = max(lift_height + 0.06, drop_pos[2] + 0.18)
    start_pos = np.array([cube_pos[0], cube_pos[1], lift_height])
    end_hover_pos = np.array([drop_pos[0], drop_pos[1], hover_drop_z])

    transfer_waypoints = _generate_arc_transfer_waypoints(
        start_pos=start_pos,
        end_pos=end_hover_pos,
        quat=quat_ny,
        arc_height=0.18,
        num_points=24,
        target_total_steps=240,
    )

    # Final descent to place the cube
    transfer_waypoints.append({"pos": [drop_pos[0], drop_pos[1], drop_pos[2]], "quat": quat_ny, "steps": 90})

    # Draw trajectory debug visualization
    debug_trajectory_lines = draw_trajectory_debug(scene, transfer_waypoints, color=(1, 1, 0, 1))

    # Get current joint state to ensure trajectory continuity
    q_current_before_transfer = _as_np(franka.get_qpos())
    
    path = generate_composite_trajectory(
        franka,
        end_effector,
        transfer_waypoints,
        default_steps=120,
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
    # Choose 0-2 random steps along the transfer to double cube mass
    base_mass = _get_mass(cube)
    mass_scale = [base_mass if base_mass is not None else None]
    mass_callbacks = {}
    if len(path) > 10:
        num_events = np.random.randint(0, 3)
        candidate_idxs = np.arange(5, len(path) - 5)
        if candidate_idxs.size > 0 and num_events > 0:
            chosen = np.random.choice(candidate_idxs, size=num_events, replace=False)

            def _make_mass_fn():
                def fn():
                    if mass_scale[0] is None:
                        return
                    new_mass = mass_scale[0] * 2.0
                    if _set_mass(cube, new_mass):
                        mass_scale[0] = new_mass
                        print(f"[Mass Change] Cube mass doubled to {new_mass:.4f}")
                return fn

            for idx in chosen:
                mass_callbacks[idx] = [_make_mass_fn()]

    execute_trajectory(
        franka,
        scene,
        cam,
        end_effector,
        cube,
        logger,
        path,
        display_video=display_video,
        step_callbacks=mass_callbacks,
        phase_name="Transport",
    )

    final_arm_qpos = path[-1][:-2]

    # Release cube at the final waypoint and stabilize
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

    # Erase trajectory debug visualization after cube is dropped
    erase_trajectory_debug(scene, debug_trajectory_lines)

    # Retreat upwards to a safe pose after placement
    retreat_qpos = franka.inverse_kinematics(
        link=end_effector,
        pos=np.array([drop_pos[0], drop_pos[1], drop_pos[2] + 0.16]),
        quat=quat_ny,
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
    spawn_area = draw_spawn_area(scene, x_range=(0.55, 0.75), y_range=(-0.28, 0.08), z_height=0.15)
    
    drop_base = np.array([0.55, 0.38, 0.14])
    drop_step = np.array([0.0, 0.0, 0.0])
    
    # Draw drop area for the first drop position
    drop_area = draw_drop_area(scene, drop_pos=drop_base, area_size=0.15, z_height=0.20)

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
