# SlipGen Genesis – Slip Prediction Dataset Generator

A complete migration of Genesis-based pick-and-place simulation with knob-driven slip generation.

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
    trajectory.py         # Trajectory generation
    trajectories.py       # Trajectory with disturbance
    steps.py              # Step-by-step execution
    demo.py               # Pick-and-place demo
    logger.py             # Sensor data + slip metrics
    data_collection.py    # Unified sensor collection
    contact_detection.py  # Contact detection
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
python run_sweep.py
```

Sweeps over:
- `μ ∈ {0.2, 0.6, 1.0}`
- `Fn_cap ∈ {2.0, 5.0, 8.0}` N
- `disturb_level ∈ {0, 1, 2}`

### Generate dataset with specific knobs
```bash
python run_dataset.py
```

Or use `run_dataset` from slipgen module:
```python
from slipgen.task_pick_place import generate_dataset
generate_dataset(
    num_samples=10,
    mu=0.6,
    fn_cap=5.0,
    disturb_level=1,
    save_path="dataset.npz"
)
```

## Legacy Code

Original source code backed up in `legacy/`:
```
legacy/
  src/              # Original src/ folder
  base_launch.py    # Original base launcher
  debug_approach.py # Original debug script
```

## Slip Detection

The `Logger` class provides:
- `detect_contact_lost()` – Contact drop event
- `get_slippage_metrics()` – Slip statistics (contact %, slip events)
- `mark_phase_start/end()` – Phase boundary tracking
- `save()` – Save sensor data to `.npz`

## Examples

Quick single-pick test (no visualization):
```python
from slipgen.scene import setup_with_knobs
from slipgen.knobs import SlipKnobs
from slipgen.logger import Logger
from slipgen.demo import run_pick_and_place_demo

knobs = SlipKnobs(mu=0.4, fn_cap=3.0, disturb_level=2)
scene, franka, cam, end_eff, cubes, motors, fingers = setup_with_knobs(knobs, show_viewer=False)
logger = Logger()
run_pick_and_place_demo(franka, scene, cam, end_eff, cubes[0], logger, motors, fingers)
print(logger.get_slippage_metrics())
```

## Notes

- Scene auto-builds after entity addition; call `scene.build()` explicitly if needed
- Force cap (Knob B) uses Genesis per-DOF force ranges where available; gracefully falls back otherwise
- Disturbance (Knob C) works by subsampling trajectory mid-segment to compress timing
- All modules are fully decoupled and importable independently
