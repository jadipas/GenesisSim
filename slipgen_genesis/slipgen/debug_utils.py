"""Debug utilities for controlling output and analyzing motion."""
import numpy as np
import sys
from io import StringIO
from contextlib import contextmanager


@contextmanager
def suppress_genesis_output():
    """Suppress verbose Genesis engine output (FPS, etc)."""
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = StringIO()
    sys.stderr = StringIO()
    try:
        yield
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr


def print_joint_state(franka, phase_name="", step_num=0):
    """Print current joint state for debugging."""
    q = franka.get_qpos()
    dq = franka.get_dofs_velocity()
    tau = franka.get_dofs_force()
    
    if hasattr(q, 'cpu'):
        q = q.cpu().numpy()
    if hasattr(dq, 'cpu'):
        dq = dq.cpu().numpy()
    if hasattr(tau, 'cpu'):
        tau = tau.cpu().numpy()
    
    print(f"\n[{phase_name} Step {step_num}]")
    print(f"  Arm joints (rad):  {np.array2string(q[:7], formatter={'float_kind': lambda x: f'{x:.4f}'})}")
    print(f"  Gripper joints:    {np.array2string(q[7:9], formatter={'float_kind': lambda x: f'{x:.4f}'})}")
    print(f"  Arm vel (rad/s):   {np.array2string(dq[:7], formatter={'float_kind': lambda x: f'{x:.4f}'})}")
    print(f"  Gripper vel:       {np.array2string(dq[7:9], formatter={'float_kind': lambda x: f'{x:.4f}'})}")
    print(f"  Max arm torque:    {np.max(np.abs(tau[:7])):.4f} N·m")
    print(f"  Max gripper force: {np.max(np.abs(tau[7:9])):.4f} N")


def print_ik_target_vs_current(franka, target_qpos, current_qpos=None, phase_name=""):
    """Print IK target vs current position and velocity changes."""
    if current_qpos is None:
        current = franka.get_qpos()
        if hasattr(current, 'cpu'):
            current = current.cpu().numpy()
    else:
        current = current_qpos
        if hasattr(current, 'cpu'):
            current = current.cpu().numpy()
        current = np.asarray(current)
    
    target = target_qpos
    if hasattr(target, 'cpu'):
        target = target.cpu().numpy()
    target = np.asarray(target)
    delta = target - current[:len(target)]
    max_delta = np.max(np.abs(delta[:7]))
    
    print(f"\n[{phase_name}] IK Target Analysis:")
    print(f"  Max arm joint delta: {max_delta:.4f} rad")
    for i, (c, t, d) in enumerate(zip(current[:7], target[:7], delta[:7])):
        print(f"    Joint {i}: {c:.4f} -> {t:.4f} (Δ={d:+.4f})")


def draw_spawn_area(scene, x_range=(0.65, 0.85), y_range=(-0.18, 0.18), z_height=0.15):
    """Draw a semi-transparent blue box for the spawn area."""
    try:
        debug_box = scene.draw_debug_box(
            bounds=[
                [x_range[0], y_range[0], 0],
                [x_range[1], y_range[1], z_height]
            ],
            color=(0, 0, 1, 0.3),
            wireframe=False,
        )
        print(f"[DEBUG] Drew spawn area box")
        return debug_box
    except Exception as e:
        print(f"[DEBUG] Failed to draw spawn area: {e}")
        return None


def draw_drop_area(scene, drop_pos, area_size=0.15, z_height=0.20):
    """Draw a semi-transparent green box for the drop-off area."""
    drop_pos = np.asarray(drop_pos, dtype=float)
    
    try:
        debug_box = scene.draw_debug_box(
            bounds=[
                [drop_pos[0] - area_size, drop_pos[1] - area_size, 0],
                [drop_pos[0] + area_size, drop_pos[1] + area_size, z_height]
            ],
            color=(0, 1, 0, 0.3),
            wireframe=False,
        )
        print(f"[DEBUG] Drew drop area box at {drop_pos}")
        return debug_box
    except Exception as e:
        print(f"[DEBUG] Failed to draw drop area: {e}")
        return None


def draw_trajectory_debug(scene, waypoints, color=(1, 1, 0, 1), line_radius=0.005):
    """Draw the intended trajectory as lines connecting waypoints."""
    debug_lines = []
    positions = []
    
    for wp in waypoints:
        pos = np.array(wp.get("pos", [np.nan, np.nan, np.nan]), dtype=float)
        if not np.any(np.isnan(pos)):
            positions.append(pos)
    
    for i in range(len(positions) - 1):
        try:
            debug_line = scene.draw_debug_line(
                start=positions[i],
                end=positions[i + 1],
                radius=line_radius,
                color=color
            )
            debug_lines.append(debug_line)
        except Exception as e:
            print(f"[DEBUG] Failed to draw trajectory line {i}: {e}")
    
    print(f"[DEBUG] Drew {len(debug_lines)} trajectory visualization lines")
    return debug_lines


def erase_trajectory_debug(scene, debug_lines):
    """Clear debug trajectory lines from the scene."""
    for debug_line in debug_lines:
        try:
            scene.clear_debug_object(debug_line)
        except Exception as e:
            print(f"[DEBUG] Failed to clear debug line: {e}")
    
    if debug_lines:
        print(f"[DEBUG] Cleared {len(debug_lines)} trajectory visualization lines")


def log_transfer_debug(waypoints, joint_path, franka, end_effector):
    """Print planned waypoints and sampled FK points to console."""
    print("[DEBUG] Transfer waypoints (Cartesian)")
    for i, wp in enumerate(waypoints):
        pos = np.array(wp.get("pos", [np.nan, np.nan, np.nan]), dtype=float)
        steps = wp.get("steps", "?")
        print(f"  [{i:02d}] pos={pos}, steps={steps}")

    fk_fn = None
    for name in ("forward_kinematics", "compute_forward_kinematics", "fk"):
        if hasattr(franka, name):
            fk_fn = getattr(franka, name)
            break

    if fk_fn is None or joint_path is None or len(joint_path) == 0:
        print("[DEBUG] FK sampling skipped (no fk fn or empty path)")
    else:
        sample_stride = max(1, len(joint_path) // 10)
        print(f"[DEBUG] Joint path samples (every {sample_stride} steps, total {len(joint_path)})")
        for idx in range(0, len(joint_path), sample_stride):
            q = joint_path[idx]
            pos = None
            for kwargs in (
                {"qpos": q, "link": end_effector},
                {"q": q, "link": end_effector},
                {"qpos": q},
            ):
                try:
                    res = fk_fn(**kwargs)
                    pos = res[0] if isinstance(res, (list, tuple)) else getattr(res, "pos", res)
                    break
                except TypeError:
                    continue
                except Exception:
                    break
            if pos is None:
                try:
                    res = fk_fn(q, end_effector)
                    pos = res[0] if isinstance(res, (list, tuple)) else getattr(res, "pos", res)
                except Exception:
                    pos = None
            if pos is None:
                continue
            pos_arr = np.asarray(pos, dtype=float)
            if pos_arr.size >= 3:
                print(f"  step {idx:04d} fk pos={pos_arr[:3]}")
