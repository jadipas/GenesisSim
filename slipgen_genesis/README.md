# SlipGen Genesis – Slip Prediction Dataset Generator

A complete migration of Genesis-based pick-and-place simulation with knob-driven slip generation and distance-based slip detection.

## Structure

```
slipgen_genesis/
  configs/
    task.yaml              # Task configuration
    curriculum.yaml        # Curriculum configuration
  slipgen/
    __init__.py
    knobs.py              # Knob definitions (μ, Fn_cap, disturb_level)
    scene.py              # Scene setup with knob wiring
    trajectory.py         # Trajectory generation & execution
    steps.py              # Step-by-step execution
    demo.py               # Pick-and-place demo
    logger.py             # Sensor data + slip metrics
    data_collection.py    # Unified sensor collection
    contact_detection.py  # Contact & slip detection
    proprioception.py     # Robot state sensing
    object_state.py       # Object pose/vel sensing
    camera.py             # Wrist camera update
    debug_utils.py        # Debug visualization
    force_viz.py          # Force plotting
  run_sweep.py            # Parameter sweep runner
  run_dataset.py          # Dataset generation runner
  test_quick.py           # Quick sanity test
```

## Three Knobs

1. **μ (contact friction)** – Applied to object material
2. **Fn_cap (max grip force)** – Finger DOF force limits
3. **disturb_level** – Transport phase acceleration bump (0..3)

## Usage

### Run a parameter sweep
```bash
cd slipgen_genesis
source ../genesis_venv/bin/activate

# Full visualization (default)
python run_sweep.py

# Headless mode (no Genesis 3D viewer)
python run_sweep.py --headless

# No OpenCV camera windows
python run_sweep.py --no-camera

# Fully headless simulation (fastest, no visualization at all)
python run_sweep.py --sim-only

# With custom number of cubes
python run_sweep.py --sim-only --num-cubes 5
```

Sweeps over:
- `μ ∈ {0.2, 0.6, 1.0}`
- `Fn_cap ∈ {2.0, 5.0, 8.0}` N
- `disturb_level ∈ {0, 1, 2}`

### Generate dataset with specific knobs
```bash
# Full visualization (default)
python run_dataset.py

# Fully headless (fastest)
python run_dataset.py --sim-only

# With all options
python run_dataset.py --sim-only \
    --num-samples 100 \
    --mu 0.4 \
    --fn-cap 3.0 \
    --disturb-level 2 \
    --output my_dataset.npz
```

### Command-line Flags

| Flag | Effect |
|------|--------|
| `--headless` | Disable Genesis 3D viewer window |
| `--no-camera` | Disable OpenCV RGB/depth camera windows |
| `--sim-only` | Both `--headless` and `--no-camera` (fastest) |

## Output File Structure (`sensor_data.npz`)

The saved `.npz` file uses a **multi-instance format** where each pick-and-place cycle (from Hover to Releasing) is stored as a separate instance. This allows for proper per-event labeling and analysis.

### Global Metadata (Scalars)
| Key | Type | Description |
|-----|------|-------------|
| `num_instances` | int | Number of pick-and-place instances in file |
| `sim_dt` | float | Simulation timestep in seconds (default: 0.01 = 100Hz) |
| `slip_threshold` | float | Distance threshold for slip detection (meters) |

### Per-Instance Data (prefixed with `instance_N_`)

Each instance `N` (0-indexed) has the following arrays. `T_N` = timesteps for that instance.

#### Instance Metadata
| Key | Type | Description |
|-----|------|-------------|
| `instance_N_timesteps` | int | Number of timesteps in instance N |
| `instance_N_first_slip_timestep` | int | First slip timestep (-1 if no slip) |

#### Slip Detection Labels
| Key | Shape | Type | Description |
|-----|-------|------|-------------|
| `instance_N_slip_mask` | (T_N,) | int32 | **Sticky** binary mask: 1 from first slip onward |
| `instance_N_slip_mask_instantaneous` | (T_N,) | int32 | **Non-sticky**: 1 only at exact slip timesteps |

#### Robot Proprioception
| Key | Shape | Type | Description |
|-----|-------|------|-------------|
| `instance_N_q` | (T_N, 9) | float | Joint positions [7 arm + 2 gripper] |
| `instance_N_dq` | (T_N, 9) | float | Joint velocities |
| `instance_N_tau` | (T_N, 9) | float | Joint torques/forces |
| `instance_N_ee_pos` | (T_N, 3) | float | End-effector position [x, y, z] (meters) |
| `instance_N_ee_quat` | (T_N, 4) | float | End-effector orientation [w, x, y, z] |
| `instance_N_ee_lin_vel` | (T_N, 3) | float | End-effector linear velocity (m/s) |
| `instance_N_gripper_width` | (T_N,) | float | Gripper opening width (meters) |

#### Object/Contact Data
| Key | Shape | Type | Description |
|-----|-------|------|-------------|
| `instance_N_in_contact` | (T_N,) | bool | Force-based contact state |
| `instance_N_cube_ee_distance` | (T_N,) | float | Cube-to-EE distance (meters) |
| `instance_N_obj_pos` | (T_N, 3) | float | Object position [x, y, z] |

### Example Usage
```python
import numpy as np

data = np.load('sensor_data.npz', allow_pickle=True)

# Global metadata
dt = float(data['sim_dt'])           # e.g., 0.01
num_instances = int(data['num_instances'])

print(f"Dataset has {num_instances} pick-and-place instances at {1/dt:.0f} Hz")

# Iterate over instances
for i in range(num_instances):
    prefix = f"instance_{i}_"
    
    T = int(data[f'{prefix}timesteps'])
    slip_mask = data[f'{prefix}slip_mask']
    first_slip = int(data[f'{prefix}first_slip_timestep'])
    
    slip_happened = first_slip >= 0
    print(f"Instance {i}: {T} steps, slip={slip_happened}")
    
    # Access sensor data for this instance
    ee_pos = data[f'{prefix}ee_pos']      # Shape: (T, 3)
    q = data[f'{prefix}q']                # Shape: (T, 9)
    labels = slip_mask                    # Shape: (T,)
```

### Inspect Dataset Tool
```bash
# Basic inspection
python inspect_dataset.py sensor_data.npz

# Inspect specific instance in detail
python inspect_dataset.py sensor_data.npz --instance 0

# Show all instances in detail
python inspect_dataset.py sensor_data.npz --verbose
```

## Slip Detection

The system uses **distance-based slip detection** which is more reliable than force-based methods in simulation:

1. **Baseline capture**: After grasp stabilization, the EE-to-cube distance is recorded
2. **Active phases**: Slip detection runs during "Lifting" and "Transport" phases
3. **Threshold check**: If displacement from baseline exceeds threshold (default: 5cm), slip is flagged
4. **Sticky mask**: Once slip occurs, all subsequent timesteps are marked as slip=1

### Logger API
```python
logger = Logger(slip_threshold=0.05, sim_dt=0.01)

# After grasp
logger.capture_grasp_baseline(ee_cube_distance)
logger.set_slip_active_phases(["Lifting", "Transport"])

# Get results
metrics = logger.get_slippage_metrics()
slip_mask = logger.get_slip_mask(sticky=True)
first_slip = logger.get_first_slip_timestep()
```

## Legacy Code

Original source code backed up in `legacy/`:
```
legacy/
  src/              # Original src/ folder
  base_launch.py    # Original base launcher
  debug_approach.py # Original debug script
```

## Examples

Quick single-pick test (no visualization):
```python
from slipgen.scene import setup_with_knobs
from slipgen.knobs import SlipKnobs
from slipgen.logger import Logger
from slipgen.demo import run_pick_and_place_demo

knobs = SlipKnobs(mu=0.4, fn_cap=3.0, disturb_level=2)
scene, franka, cam, end_eff, cubes, motors, fingers = setup_with_knobs(knobs, show_viewer=False)
logger = Logger(sim_dt=0.01)
run_pick_and_place_demo(franka, scene, cam, end_eff, cubes[0], logger, motors, fingers,
                        show_viewer=False, render_cameras=False)
print(logger.get_slippage_metrics())
```

## Notes

- Scene auto-builds after entity addition; call `scene.build()` explicitly if needed
- Force cap (Knob B) uses Genesis per-DOF force ranges where available; gracefully falls back otherwise
- Disturbance (Knob C) works by subsampling trajectory mid-segment to compress timing
- All modules are fully decoupled and importable independently
- Timesteps are in simulation steps (not seconds); multiply by `sim_dt` to get time
