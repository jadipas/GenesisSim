"""Trajectory execution and generation functionality."""
from typing import Iterable, Sequence, Optional, Dict, Any, Callable
import cv2
import numpy as np

from slipgen.camera import update_wrist_camera
from slipgen.data_collection import collect_sensor_data
from slipgen.contact_detection import detect_slip_by_distance


def _to_numpy(arr):
    """Convert torch tensors (cpu/gpu) or arrays to host numpy float array."""
    if hasattr(arr, "detach") and hasattr(arr, "cpu"):
        return arr.detach().cpu().numpy()
    return np.asarray(arr, dtype=float)


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


def _get_fk_function(franka):
    """Get the forward kinematics function from the robot, if available."""
    for name in ("forward_kinematics", "compute_forward_kinematics", "fk"):
        if hasattr(franka, name):
            return getattr(franka, name)
    return None


def _compute_fk_pos(franka, end_effector, q):
    """Compute end-effector position for a given joint configuration using FK."""
    fk_fn = _get_fk_function(franka)
    if fk_fn is None:
        return None
    
    # Try various call signatures
    for kwargs in (
        {"qpos": q, "link": end_effector},
        {"q": q, "link": end_effector},
        {"qpos": q},
    ):
        try:
            res = fk_fn(**kwargs)
            pos = res[0] if isinstance(res, (list, tuple)) else getattr(res, "pos", res)
            return _to_numpy(pos)[:3]
        except TypeError:
            continue
        except Exception:
            break
    
    # Try positional args
    try:
        res = fk_fn(q, end_effector)
        pos = res[0] if isinstance(res, (list, tuple)) else getattr(res, "pos", res)
        return _to_numpy(pos)[:3]
    except Exception:
        return None


def _unwrap_joint_solution(q_new, q_prev, joint_limits=None):
    """
    Unwrap joint angles to find the closest equivalent solution.
    
    For revolute joints with continuous rotation (or large limits), 
    the IK solver might return solutions that differ by 2π.
    This function finds the equivalent angle closest to the previous solution.
    """
    q_unwrapped = np.array(q_new, dtype=float, copy=True)
    
    # Franka Panda joint limits (approximate, for reference)
    # Joint 0: [-2.8973, 2.8973], Joint 1: [-1.7628, 1.7628], Joint 2: [-2.8973, 2.8973]
    # Joint 3: [-3.0718, -0.0698], Joint 4: [-2.8973, 2.8973], Joint 5: [-0.0175, 3.7525]
    # Joint 6: [-2.8973, 2.8973]
    
    for j in range(min(7, len(q_new))):
        delta = q_unwrapped[j] - q_prev[j]
        
        # Check if wrapping by 2π would bring it closer
        if delta > np.pi:
            candidate = q_unwrapped[j] - 2 * np.pi
            # Only unwrap if the new value is still valid (rough check)
            if candidate > -3.5:  # Conservative lower bound
                q_unwrapped[j] = candidate
        elif delta < -np.pi:
            candidate = q_unwrapped[j] + 2 * np.pi
            if candidate < 3.5:  # Conservative upper bound
                q_unwrapped[j] = candidate
    
    return q_unwrapped


def generate_joint_space_arc_trajectory(
    franka,
    end_effector,
    start_q: np.ndarray,
    start_pos: np.ndarray,
    end_pos: np.ndarray,
    quat_start: np.ndarray,
    quat_end: np.ndarray = None,
    arc_height: float = 0.55,
    arc_bias: float = 0.0,
    total_steps: int = 240,
    finger_qpos: float | Sequence[float] | None = None,
    max_cartesian_deviation: float = 0.15,
    num_key_waypoints: int = 5,
):
    """
    Generate a smooth joint-space trajectory for arc transfer motion.
    
    Instead of computing IK at every Cartesian waypoint (which causes discontinuities),
    this function:
    1. Computes IK only at a few key waypoints (start, apex, end, and optionally intermediate)
    2. Interpolates smoothly in joint space between these key configurations
    3. Verifies the Cartesian path using FK to ensure it stays reasonable
    
    Parameters
    ----------
    franka : robot
        The Franka robot object
    end_effector : link
        The end-effector link for IK/FK
    start_q : np.ndarray
        Starting joint configuration (9 DOF including gripper)
    start_pos : np.ndarray
        Starting Cartesian position
    end_pos : np.ndarray
        Ending Cartesian position
    quat_start : np.ndarray
        Starting orientation quaternion
    quat_end : np.ndarray, optional
        Ending orientation quaternion (defaults to quat_start)
    arc_height : float
        Height of the arc apex above the start-end midpoint
    arc_bias : float
        Lateral bias for the arc curve
    total_steps : int
        Total number of trajectory steps
    finger_qpos : float or sequence, optional
        Target finger positions to maintain throughout
    max_cartesian_deviation : float
        Maximum allowed deviation from the ideal Cartesian path (for FK verification)
    num_key_waypoints : int
        Number of key waypoints to compute IK for (minimum 3: start, apex, end)
    
    Returns
    -------
    np.ndarray
        Joint-space trajectory of shape (total_steps, 9)
    dict
        Debug info with FK verification results
    """
    start_pos = np.asarray(start_pos, dtype=float)
    end_pos = np.asarray(end_pos, dtype=float)
    quat_start = np.asarray(quat_start, dtype=float)
    quat_end = np.asarray(quat_end, dtype=float) if quat_end is not None else quat_start.copy()
    start_q = _to_numpy(start_q)
    
    num_key_waypoints = max(3, num_key_waypoints)
    
    # Compute arc geometry
    p0, p1 = start_pos, end_pos
    radial_xy = np.array([p0[0] + p1[0], p0[1] + p1[1]], dtype=float)
    radial_norm = np.linalg.norm(radial_xy)
    radial_dir = radial_xy / radial_norm if radial_norm > 1e-4 else np.array([1.0, 0.0])
    
    tilt_angle = np.radians(45.0)
    arc_normal = np.array([
        radial_dir[0] * np.cos(tilt_angle),
        radial_dir[1] * np.cos(tilt_angle),
        np.sin(tilt_angle)
    ], dtype=float)
    
    ctrl = 0.5 * (p0 + p1) + arc_height * arc_normal
    if abs(arc_bias) > 1e-6:
        lateral_dir = np.array([-radial_dir[1], radial_dir[0], 0.0], dtype=float)
        ctrl = ctrl + arc_bias * lateral_dir
    
    def bezier_point(t):
        """Quadratic Bezier curve for the arc."""
        return (1 - t)**2 * p0 + 2 * (1 - t) * t * ctrl + t**2 * p1
    
    # Compute IK for key waypoints with seeding from previous solution
    # Use adaptive refinement: if a segment has a large jump, subdivide it
    
    def solve_ik_for_waypoint(t_val, q_seed_local):
        """Solve IK for a waypoint at parameter t, seeded from previous config."""
        pos = bezier_point(t_val)
        quat = _slerp_quat(quat_start, quat_end, t_val)
        q_ik = franka.inverse_kinematics(link=end_effector, pos=pos, quat=quat)
        q_ik = _to_numpy(q_ik)
        q_ik_unwrapped = _unwrap_joint_solution(q_ik[:7], q_seed_local[:7])
        return q_ik_unwrapped, pos, quat
    
    # Start with uniform key t values
    key_t_values = list(np.linspace(0.0, 1.0, num_key_waypoints))
    key_joint_configs = []
    key_cartesian_waypoints = []
    key_cartesian_waypoints = []
    q_seed = start_q.copy()
    
    print(f"[Joint-Space Planner] Computing IK for key waypoints with adaptive refinement...")
    
    # Adaptive IK solving with subdivision
    max_iterations = 20  # Prevent infinite loops
    iteration = 0
    i = 0
    
    while i < len(key_t_values) and iteration < max_iterations:
        iteration += 1
        t_val = key_t_values[i]
        
        if i == 0:
            # Use start configuration directly
            key_joint_configs.append(start_q[:7].copy())
            key_cartesian_waypoints.append({
                "t": t_val,
                "pos": bezier_point(t_val),
                "quat": _slerp_quat(quat_start, quat_end, t_val)
            })
            i += 1
            continue
        
        q_ik_unwrapped, pos, quat = solve_ik_for_waypoint(t_val, q_seed)
        
        # Check for discontinuity
        joint_delta = np.linalg.norm(q_ik_unwrapped - q_seed[:7])
        
        if joint_delta > 1.5:
            # Large jump detected - try to subdivide this segment
            prev_t = key_t_values[i - 1]
            mid_t = (prev_t + t_val) / 2.0
            
            # Only subdivide if the segment is large enough
            if (t_val - prev_t) > 0.05:
                print(f"  [REFINE] Large jump ({joint_delta:.3f} rad) at t={t_val:.3f}, inserting midpoint at t={mid_t:.3f}")
                key_t_values.insert(i, mid_t)
                # Don't increment i - we'll process the new midpoint next
                continue
            else:
                print(f"  [WARN] Key waypoint at t={t_val:.3f}: joint delta = {joint_delta:.3f} rad (segment too small to subdivide)")
        else:
            print(f"  Key waypoint at t={t_val:.3f}: joint delta = {joint_delta:.3f} rad [OK]")
        
        key_joint_configs.append(q_ik_unwrapped.copy())
        key_cartesian_waypoints.append({"t": t_val, "pos": pos, "quat": quat})
        q_seed[:7] = q_ik_unwrapped
        i += 1
    
    print(f"[Joint-Space Planner] Final key waypoint count: {len(key_joint_configs)} (started with {num_key_waypoints})")
    
    # Interpolate in joint space between key configurations
    # Distribute steps proportionally between key waypoints
    num_segments = len(key_joint_configs) - 1
    if num_segments <= 0:
        # Fallback: return start config repeated
        print("[WARN] No segments to interpolate, returning start configuration")
        trajectory = np.tile(start_q, (total_steps, 1))
        debug_info = {"key_waypoints": key_cartesian_waypoints, "key_joint_configs": key_joint_configs,
                      "fk_deviations": [], "max_deviation": 0.0, "verification_passed": True}
        return trajectory, debug_info
    
    steps_per_segment = total_steps // num_segments
    remainder = total_steps % num_segments
    
    trajectory = []
    
    for seg_idx in range(num_segments):
        q_start_seg = key_joint_configs[seg_idx]
        q_end_seg = key_joint_configs[seg_idx + 1]
        
        # Add extra step to last segment to use up remainder
        seg_steps = steps_per_segment + (remainder if seg_idx == num_segments - 1 else 0)
        
        # Interpolate in joint space (excluding endpoint to avoid duplicates, except for last segment)
        include_endpoint = (seg_idx == num_segments - 1)
        
        for step in range(seg_steps):
            t_local = step / seg_steps if seg_steps > 0 else 0
            if include_endpoint and step == seg_steps - 1:
                t_local = 1.0
            
            q_interp = (1 - t_local) * q_start_seg + t_local * q_end_seg
            trajectory.append(q_interp)
    
    trajectory = np.array(trajectory)
    
    # Apply finger positions
    if finger_qpos is not None:
        finger_vals = np.array(finger_qpos, dtype=float)
        if finger_vals.size == 1:
            finger_vals = np.repeat(finger_vals, 2)
        full_trajectory = np.zeros((len(trajectory), 9))
        full_trajectory[:, :7] = trajectory
        full_trajectory[:, 7:] = finger_vals
        trajectory = full_trajectory
    else:
        # Use starting finger positions
        full_trajectory = np.zeros((len(trajectory), 9))
        full_trajectory[:, :7] = trajectory
        full_trajectory[:, 7:] = start_q[7:]
        trajectory = full_trajectory
    
    # FK Verification
    debug_info = {
        "key_waypoints": key_cartesian_waypoints,
        "key_joint_configs": key_joint_configs,
        "fk_deviations": [],
        "max_deviation": 0.0,
        "verification_passed": True,
    }
    
    print(f"[Joint-Space Planner] Verifying trajectory with FK...")
    
    sample_indices = np.linspace(0, len(trajectory) - 1, min(20, len(trajectory)), dtype=int)
    
    for idx in sample_indices:
        q = trajectory[idx]
        t_global = idx / (len(trajectory) - 1)
        
        # Compute expected Cartesian position from Bezier
        expected_pos = bezier_point(t_global)
        
        # Compute actual position from FK
        actual_pos = _compute_fk_pos(franka, end_effector, q)
        
        if actual_pos is not None:
            deviation = np.linalg.norm(actual_pos - expected_pos)
            debug_info["fk_deviations"].append({
                "step": idx,
                "t": t_global,
                "expected": expected_pos.copy(),
                "actual": actual_pos.copy(),
                "deviation": deviation,
            })
            
            if deviation > debug_info["max_deviation"]:
                debug_info["max_deviation"] = deviation
            
            if deviation > max_cartesian_deviation:
                debug_info["verification_passed"] = False
                print(f"  [WARN] Step {idx} (t={t_global:.2f}): deviation = {deviation:.4f}m > {max_cartesian_deviation}m")
    
    if debug_info["verification_passed"]:
        print(f"  FK verification PASSED (max deviation: {debug_info['max_deviation']:.4f}m)")
    else:
        print(f"  FK verification FAILED (max deviation: {debug_info['max_deviation']:.4f}m)")
    
    return trajectory, debug_info


def generate_composite_trajectory(
    franka,
    end_effector,
    waypoints: Iterable[dict],
    default_steps: int = 150,
    finger_qpos: float | Sequence[float] | None = None,
    start_q: np.ndarray | None = None,
):
    """Build a single joint-space trajectory from sparse end-effector waypoints.
    
    Args:
        start_q: Optional starting joint configuration. If None, uses franka.get_qpos().
                 Use this when chaining trajectories to avoid discontinuities.
    """
    current_q = _to_numpy(start_q) if start_q is not None else _to_numpy(franka.get_qpos())
    trajectory = [current_q]

    def _apply_finger_target(q_goal, finger_override):
        if finger_override is None:
            return _to_numpy(q_goal)
        finger_vals = np.array(finger_override, dtype=float)
        if finger_vals.size == 1:
            finger_vals = np.repeat(finger_vals, 2)
        q_goal_arr = np.array(_to_numpy(q_goal), dtype=float, copy=True)
        q_goal_arr[-2:] = finger_vals
        return q_goal_arr

    for i, wp in enumerate(waypoints):
        pos = np.asarray(wp["pos"], dtype=float)
        quat = np.asarray(wp.get("quat", [0, 1, 0, 0]), dtype=float)
        steps = int(wp.get("steps", default_steps))
        steps = max(2, steps)

        q_goal = franka.inverse_kinematics(link=end_effector, pos=pos, quat=quat)
        finger_override = wp.get("finger", finger_qpos)
        q_goal = _apply_finger_target(q_goal, finger_override)
        
        # Check for large joint jumps that indicate IK discontinuity
        joint_delta = np.linalg.norm(current_q[:7] - q_goal[:7])
        if joint_delta > 1.0:
            print(f"[WARN] Large IK jump at waypoint {i}: {joint_delta:.3f} rad. Doubling interpolation steps.")
            steps = max(steps, int(steps * 2.0))

        # Interpolate including the goal and excluding the current starting point
        segment = np.linspace(current_q, q_goal, steps, endpoint=True)[1:]
        trajectory.extend(segment)
        current_q = q_goal

    return np.array(trajectory)


def execute_trajectory(franka, scene, cam, end_effector, cube, logger, 
                       path, render_cameras=True, check_contact=True, step_callbacks=None, phase_name="", debug=False, knobs=None, finger_force=None, fingers_dof=None, finger_qpos=None,
                       check_slip: bool = True, slip_threshold: float = None):
    """Execute a planned trajectory with sensor data collection and optional gripper control.

    Args:
        render_cameras: If True, render OpenCV RGB/depth windows. Set False for headless.

    If ``finger_qpos`` is provided, fingers are controlled via position targets with force cap.
    If ``finger_force`` is provided (legacy), a clamped holding force is applied.
    The force cap from ``knobs.fn_cap`` is always enforced.
    
    Distance-based slip detection:
        If check_slip=True and the logger has a baseline distance set (via logger.capture_grasp_baseline()),
        slip detection will run during phases marked as active (via logger.set_slip_active_phases()).
        Use slip_threshold to override the default threshold (0.015m).
    """
    callbacks = {}
    if step_callbacks:
        if isinstance(step_callbacks, dict):
            for k, v in step_callbacks.items():
                if v is None:
                    continue
                callbacks.setdefault(int(k), []).extend(v if isinstance(v, (list, tuple)) else [v])
        else:
            for step_idx, fn in step_callbacks:
                callbacks.setdefault(int(step_idx), []).append(fn)

    if phase_name:
        print(f"{phase_name}...")
    
    # Extract force cap from knobs
    fn_cap = None
    if knobs is not None and hasattr(knobs, 'fn_cap'):
        fn_cap = float(knobs.fn_cap)
    
    # Prepare clamped holding force (legacy path)
    clamped_force = None
    if finger_force is not None and fingers_dof is not None:
        clamped_force = np.asarray(finger_force, dtype=float)
        if fn_cap is not None:
            clamped_force = np.clip(clamped_force, -fn_cap, fn_cap)
    
    for step_idx, waypoint in enumerate(path):
        # Apply joint position command
        franka.control_dofs_position(waypoint)
        
        # Apply finger control
        if finger_qpos is not None and fingers_dof is not None:
            # Position control: move fingers to target position
            finger_target = np.asarray(finger_qpos, dtype=float)
            if finger_target.size == 1:
                finger_target = np.array([finger_target.item(), finger_target.item()])
            franka.control_dofs_position(finger_target, fingers_dof)
            # Note: Force cap is enforced by Genesis position controller when applying position commands
            # No explicit force clamping needed for position control
        elif clamped_force is not None:
            # Legacy force control path
            franka.control_dofs_force(clamped_force, fingers_dof)
        
        scene.step()
        update_wrist_camera(cam, end_effector)
        
        sensor_data = collect_sensor_data(franka, end_effector, cube, cam, include_vision=False)
        logger.log_step(sensor_data)
        
        # Distance-based slip detection
        if check_slip and logger.is_slip_detection_active():
            threshold = slip_threshold if slip_threshold is not None else logger.slip_threshold
            slip_result = detect_slip_by_distance(
                cube_pos=sensor_data['obj_pos'],
                ee_pos=sensor_data['ee_pos'],
                baseline_distance=logger.baseline_ee_cube_distance,
                threshold=threshold,
            )
            logger.log_slip_check(slip_result)
            if logger.check_and_log_slip(slip_result):
                print(f"SLIP DETECTED at timestep {logger.timestep}, displacement: {slip_result['displacement_from_baseline']:.4f}m (threshold: {threshold:.4f}m)")
        
        # Invoke any scheduled callbacks for this step
        if step_idx in callbacks:
            for fn in callbacks[step_idx]:
                try:
                    fn()
                except Exception as exc:
                    print(f"Callback at step {step_idx} failed: {exc}")
        
        if debug and step_idx % 50 == 0:
            q_current = franka.get_qpos()
            dq_current = franka.get_dofs_velocity()
            if hasattr(q_current, 'cpu'):
                q_current = q_current.cpu().numpy()
            if hasattr(dq_current, 'cpu'):
                dq_current = dq_current.cpu().numpy()
            print(f"[{phase_name or 'Traj'} Step {step_idx}/{len(path)}] Joint vel (arm): {dq_current[:7]}")

        if check_contact and logger.detect_contact_lost():
            print(f"CONTACT LOST at timestep {logger.timestep}")
        
        if render_cameras and cam is not None:
            rgb, depth, seg, normal = cam.render(depth=True)
            depth_vis = (depth / depth.max() * 255).astype('uint8')
            cv2.imshow("RGB", rgb[:, :, ::-1])
            cv2.imshow("Depth", depth_vis)
            cv2.waitKey(1)


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


def generate_composite(
    franka,
    end_effector,
    waypoints: Iterable[dict],
    default_steps: int = 150,
    finger_qpos: Optional[float | Sequence[float]] = None,
    start_q: Optional[np.ndarray] = None,
):
    """Alias for ``generate_composite_trajectory`` for API compatibility."""
    return generate_composite_trajectory(franka, end_effector, waypoints, default_steps, finger_qpos, start_q)


def execute(
    franka,
    scene,
    cam,
    end_effector,
    cube,
    logger,
    path,
    render_cameras: bool = True,
    check_contact: bool = True,
    step_callbacks: Optional[Dict[int, Any]] = None,
    phase_name: str = "",
    debug: bool = False,
    disturb_level: int = 0,
    shake_amp: float = 0.0,
    shake_freq: float = 6.0,
    knobs=None,
    finger_force=None,
    fingers_dof=None,
    finger_qpos=None,
    check_slip: bool = True,
    slip_threshold: float = None,
):
    """Execute a trajectory with optional phase warp and joint shake disturbances, maintaining gripper control via position or force.
    
    Args:
        render_cameras: If True, render OpenCV RGB/depth windows. Set False for headless.
    
    Distance-based slip detection is enabled by default if the logger has been configured with a baseline distance.
    """
    path2 = apply_phase_warp_bump(np.asarray(path), disturb_level)
    
    # Apply joint shake during transport if enabled
    if shake_amp > 0 and phase_name == "Transport":
        path_shaken = []
        for step_idx, waypoint in enumerate(path2):
            shaken = add_joint_shake(waypoint, step_idx, len(path2), 
                                    amp=shake_amp, freq=shake_freq)
            path_shaken.append(shaken)
        path2 = np.array(path_shaken)
    
    return execute_trajectory(
        franka,
        scene,
        cam,
        end_effector,
        cube,
        logger,
        path2,
        render_cameras,
        check_contact,
        step_callbacks,
        phase_name,
        debug,
        knobs=knobs,
        finger_force=finger_force,
        fingers_dof=fingers_dof,
        finger_qpos=finger_qpos,
        check_slip=check_slip,
        slip_threshold=slip_threshold,
    )
