"""Pick-and-place demonstration functionality."""
import numpy as np
from .trajectory import execute_trajectory, generate_composite_trajectory
from .steps import execute_steps
from src.utils import print_ik_target_vs_current


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
    print(f"[DEBUG] Hover target: pos={hover_target_pos}, quat={quat_ny}")
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

    # Plan and execute transport to drop site
    mid_xy = np.array([(cube_pos[0] + drop_pos[0]) / 2, (cube_pos[1] + drop_pos[1]) / 2])
    hover_drop_z = max(lift_height + 0.06, drop_pos[2] + 0.18)
    apex_z = hover_drop_z + 0.18  # higher arc
    transfer_waypoints = [
        {"pos": [cube_pos[0], cube_pos[1], lift_height], "quat": quat_ny, "steps": 120},
        {"pos": [mid_xy[0], mid_xy[1], apex_z], "quat": quat_ny, "steps": 120},
        {"pos": [drop_pos[0], drop_pos[1], hover_drop_z], "quat": quat_ny, "steps": 100},
        {"pos": [drop_pos[0], drop_pos[1], drop_pos[2]], "quat": quat_ny, "steps": 90},
    ]

    path = generate_composite_trajectory(
        franka,
        end_effector,
        transfer_waypoints,
        default_steps=120,
        finger_qpos=0.0,
    )
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

    drop_base = np.array([0.55, 0.38, 0.14])
    drop_step = np.array([0.0, -0.08, 0.0])

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
        )

        # Despawn placed cube
        try:
            scene.remove_entity(cube)
        except Exception:
            cube.set_pos(np.array([2.0 + i, 2.0, 0.05]))
        print(f"Cube {i+1} placed and removed.")
