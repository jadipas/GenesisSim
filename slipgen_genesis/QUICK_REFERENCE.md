# Quick Reference: Transport Phase Improvements

## What Changed (Visual Summary)

```
BEFORE (BROKEN):
├─ Timing Bug: target_total_steps=24 → ~92 steps total (under-sampled!)
├─ _apply_accel_bump: Drops samples → IK jumps, controller artifacts
├─ No explicit disturbance controls
└─ Implicit parameters buried in code

AFTER (FIXED):
├─ Fixed: target_total_steps=240 → proper resolution
├─ apply_phase_warp_bump: Smooth time warping, no dropped samples
├─ add_joint_shake: Clean sinusoidal disturbances
├─ arc_bias: Lateral curvature for tangential stress
└─ Explicit parameters exposed in function signatures
```

## Function Signatures Changed

### `run_pick_and_place_demo()`

```python
# OLD
def run_pick_and_place_demo(
    franka, scene, cam, end_effector, cube, logger,
    motors_dof, fingers_dof,
    display_video=True,
    drop_pos=None,
    debug_plot_transfer=True,
)

# NEW - Added 4 parameters
def run_pick_and_place_demo(
    franka, scene, cam, end_effector, cube, logger,
    motors_dof, fingers_dof,
    display_video=True,
    drop_pos=None,
    debug_plot_transfer=True,
    transport_steps=240,      # ← NEW: explicit duration
    phase_warp_level=0,       # ← NEW: smooth speedup (0-3)
    shake_amp=0.0,            # ← NEW: joint shake amplitude
    shake_freq=6.0,           # ← NEW: shake frequency
)
```

### `run_iterative_pick_and_place()`

```python
# Same 4 new parameters added here too
```

## Usage (Copy-Paste Ready)

### Minimal change (maintain old behavior):
```python
run_pick_and_place_demo(
    franka, scene, cam, end_effector, cube, logger,
    motors_dof, fingers_dof,
    display_video=True,
    drop_pos=my_pos,
    # Defaults provide same behavior as before (but fixed)
)
```

### With disturbances:
```python
run_pick_and_place_demo(
    franka, scene, cam, end_effector, cube, logger,
    motors_dof, fingers_dof,
    display_video=True,
    drop_pos=my_pos,
    transport_steps=240,
    phase_warp_level=2,    # Medium speedup in middle
    shake_amp=0.02,        # 1.15° shake
    shake_freq=6.0,        # 6 cycles
)
```

## Parameter Cheat Sheet

| Parameter | Type | Range | Effect |
|-----------|------|-------|--------|
| `transport_steps` | int | 180-320 | Lower = faster transport |
| `phase_warp_level` | int | 0-3 | 0=none, 1=mild, 2=med, 3=strong speedup |
| `shake_amp` | float | 0.0-0.03 | Joint oscillation amplitude (radians) |
| `shake_freq` | float | 4.0-8.0 | Cycles over full trajectory |

### Quick Recipes:

```python
# No disturbance (baseline)
transport_steps=240, phase_warp_level=0, shake_amp=0.0

# Light stress
transport_steps=240, phase_warp_level=1, shake_amp=0.01

# Medium stress  
transport_steps=240, phase_warp_level=2, shake_amp=0.02

# Heavy stress
transport_steps=200, phase_warp_level=3, shake_amp=0.03

# Slow and gentle
transport_steps=320, phase_warp_level=0, shake_amp=0.0
```

## What Each Disturbance Does

### `phase_warp_level` (smooth acceleration)
- Creates "speedup" in the middle of transport
- Keeps total steps constant (no controller timing issues)
- Uses smooth tanh-based phase mapping
- Level 1: 35% compression → gentle
- Level 2: 65% compression → moderate  
- Level 3: 100% compression → aggressive

**Physical effect**: Higher tangential velocity → more inertial demand → slip risk

### `shake_amp` + `shake_freq` (oscillation)
- Adds sinusoidal perturbations to joints 3 & 5
- Amplitude in radians (0.01 ≈ 0.57°, 0.03 ≈ 1.72°)
- Frequency = number of complete cycles over trajectory
- Only active during "Transport" phase

**Physical effect**: Lateral accelerations → tangential force spikes → slip risk

### `arc_bias` (lateral curvature) - auto-randomized
- Already randomized ±0.03m per trajectory
- Shifts arc control point sideways
- Creates consistent curved path

**Physical effect**: Continuous lateral acceleration → sustained tangential stress

## Internal Changes (for reference)

### New functions in `demo.py`:
- `apply_phase_warp_bump(path, level)` - smooth time warping
- `add_joint_shake(q, step_idx, total_steps, amp, freq, joints)` - sinusoidal perturbation

### Updated functions in `demo.py`:
- `_generate_arc_transfer_waypoints()` - added `arc_bias` parameter
- `run_pick_and_place_demo()` - 4 new parameters
- `run_iterative_pick_and_place()` - 4 new parameters

### Changes in `trajectory.py`:
- Replaced `_apply_accel_bump()` with `apply_phase_warp_bump()`
- Added `add_joint_shake()`
- Updated `execute()` with `shake_amp` and `shake_freq` parameters

## Migration Checklist

- [ ] Update function calls to include new parameters (or use defaults)
- [ ] Remove any manual `target_total_steps=24` if present elsewhere
- [ ] Test baseline config (should match old behavior but smoother)
- [ ] Test disturbance configs (should see slip events)
- [ ] Update sweep/dataset scripts with parameter sweeps
- [ ] Verify no IK jumps during transport (check logs)

## Expected Behavior

✅ **Correct:**
- Transport takes ~240 steps (or specified `transport_steps`)
- Smooth joint trajectories (no sudden jumps)
- Optional speedup in middle (if `phase_warp_level > 0`)
- Optional small oscillations (if `shake_amp > 0`)
- Slip events correlated with disturbance magnitude

❌ **Incorrect (old bug):**
- ~~Transport takes ~92 steps~~ → FIXED
- ~~Large IK jumps at waypoints~~ → FIXED  
- ~~Controller struggling to track~~ → FIXED
- ~~Inconsistent slip behavior~~ → FIXED

## Questions?

See `TRANSPORT_IMPROVEMENTS.md` for detailed explanations.
See `examples_disturbance_configs.py` for usage examples and sweep strategies.
