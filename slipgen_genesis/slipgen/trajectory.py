"""Trajectory execution and generation functionality."""
from typing import Iterable, Sequence, Optional, Dict, Any
import cv2
import numpy as np

from slipgen.camera import update_wrist_camera
from slipgen.data_collection import collect_sensor_data


def _to_numpy(arr):
    """Convert torch tensors (cpu/gpu) or arrays to host numpy float array."""
    if hasattr(arr, "detach") and hasattr(arr, "cpu"):
        return arr.detach().cpu().numpy()
    return np.asarray(arr, dtype=float)


def generate_composite_trajectory(
    franka,
    end_effector,
    waypoints: Iterable[dict],
    default_steps: int = 150,
    finger_qpos: float | Sequence[float] | None = None,
):
    """Build a single joint-space trajectory from sparse end-effector waypoints."""
    current_q = _to_numpy(franka.get_qpos())
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
                       path, display_video=True, check_contact=True, step_callbacks=None, phase_name="", debug=False, knobs=None, finger_force=None, fingers_dof=None):
    """Execute a planned trajectory with sensor data collection and optional gripper force maintenance."""
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
    
    for step_idx, waypoint in enumerate(path):
        # Apply joint shake during Transport phase if disturbance is enabled
        # Note: shake parameters should be passed via step_callbacks or similar if needed
        franka.control_dofs_position(waypoint)
        scene.step()
        update_wrist_camera(cam, end_effector)
        
        sensor_data = collect_sensor_data(franka, end_effector, cube, cam, include_vision=False)
        logger.log_step(sensor_data)
        
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
        
        if display_video and cam is not None:
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
):
    """Alias for ``generate_composite_trajectory`` for API compatibility."""
    return generate_composite_trajectory(franka, end_effector, waypoints, default_steps, finger_qpos)


def execute(
    franka,
    scene,
    cam,
    end_effector,
    cube,
    logger,
    path,
    display_video: bool = True,
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
):
    """Execute a trajectory with optional phase warp and joint shake disturbances, maintaining gripper force."""
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
        display_video,
        check_contact,
        step_callbacks,
        phase_name,
        debug,
        knobs=knobs,
        finger_force=finger_force,
        fingers_dof=fingers_dof,
    )
