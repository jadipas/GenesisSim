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


def check_joint_limits_violated(franka, lower_limits, upper_limits):
    """Check if any joint position violates its limits."""
    q = franka.get_qpos()
    if hasattr(q, 'cpu'):
        q = q.cpu().numpy()
    
    violations = []
    for i, (q_i, lower, upper) in enumerate(zip(q, lower_limits, upper_limits)):
        if q_i < lower or q_i > upper:
            violations.append(f"Joint {i}: {q_i:.4f} (limits: [{lower:.4f}, {upper:.4f}])")
    
    if violations:
        print("[WARNING] Joint limits violated:")
        for v in violations:
            print(f"  - {v}")
        return True
    return False


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
    max_delta = np.max(np.abs(delta[:7]))  # arm only
    
    print(f"\n[{phase_name}] IK Target Analysis:")
    print(f"  Max arm joint delta: {max_delta:.4f} rad")
    for i, (c, t, d) in enumerate(zip(current[:7], target[:7], delta[:7])):
        print(f"    Joint {i}: {c:.4f} -> {t:.4f} (Δ={d:+.4f})")
