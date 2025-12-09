"""Trajectory execution and generation functionality."""
from typing import Iterable, Sequence

import cv2
import numpy as np

from src.camera import update_wrist_camera
from src.sensors import collect_sensor_data


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
    """
    Build a single joint-space trajectory from sparse end-effector waypoints.

    Each waypoint should be a dict with:
    - ``pos``: iterable of 3 floats (x, y, z)
    - ``quat``: iterable of 4 floats (w, x, y, z)
    - optional ``steps``: int number of interpolation steps for this segment
    - optional ``finger``: float or length-2 iterable to override finger joints

    Args:
        franka: Robot entity
        end_effector: End-effector link
        waypoints: Sequence of waypoint dictionaries
        default_steps: Fallback step count between waypoints
        finger_qpos: Default finger joint command (scalar duplicated to 2 joints
            or length-2 iterable). If ``None``, use IK result for fingers.

    Returns:
        np.ndarray of shape (N, 9) with concatenated joint positions.
    """
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

    for wp in waypoints:
        pos = np.asarray(wp["pos"], dtype=float)
        quat = np.asarray(wp.get("quat", [0, 1, 0, 0]), dtype=float)
        steps = int(wp.get("steps", default_steps))
        steps = max(2, steps)

        q_goal = franka.inverse_kinematics(link=end_effector, pos=pos, quat=quat)
        finger_override = wp.get("finger", finger_qpos)
        q_goal = _apply_finger_target(q_goal, finger_override)

        # Interpolate including the goal and excluding the current starting point
        segment = np.linspace(current_q, q_goal, steps, endpoint=True)[1:]
        trajectory.extend(segment)
        current_q = q_goal

    return np.array(trajectory)


def execute_trajectory(franka, scene, cam, end_effector, cube, logger, 
                       path, display_video=True, check_contact=True, step_callbacks=None, phase_name="", debug=False):
    """
    Execute a planned trajectory with sensor data collection.
    
    Args:
        franka: Robot entity
        scene: Simulation scene
        cam: Camera object
        end_effector: End-effector link
        cube: Cube entity
        logger: SensorDataLogger instance
        path: Trajectory waypoints
        display_video: Whether to display RGB/Depth video
        check_contact: Whether to check for contact lost events
        step_callbacks: Optional callbacks to invoke at specific steps
        phase_name: Name of phase for logging
        debug: Whether to print detailed debug info
    """
    callbacks = {}
    if step_callbacks:
        # Normalize to dict of step index -> list of callables
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
