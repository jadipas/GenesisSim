# Genesis Franka Pick-and-Place Simulation

A modular simulation framework for robotic pick-and-place tasks using the Genesis physics engine and Franka Emika Panda robot. This project includes sensor data collection, contact detection, and automated labeling for slip detection research.

## Overview

This simulation framework enables:
- **Robotic manipulation** with the Franka Emika Panda arm
- **Sensor data collection** including proprioception, contact forces, and vision
- **Ground-truth contact labeling** for machine learning datasets
- **Wrist-mounted camera** simulation with RGB-D capabilities
- **Modular architecture** for easy extension and research

## Project Structure

```
GenesisSim/
├── base_launch.py          # Main entry point
├── sensor_data.npz         # Collected sensor data output
├── xml/                    # Robot MJCF files
│   └── franka_emika_panda/
└── src/                    # Source modules
    ├── config/             # Configuration and initialization
    │   ├── args.py         # Command-line argument parsing
    │   └── init_sim.py     # Genesis simulator initialization
    │
    ├── sensors/            # Sensor data collection
    │   ├── logger.py       # SensorDataLogger class
    │   ├── contact_detection.py    # Contact detection logic
    │   ├── proprioception.py       # Joint state sensing
    │   ├── object_state.py         # Object pose tracking
    │   └── data_collection.py      # Unified data collection
    │
    ├── scene/              # Scene setup
    │   ├── setup.py        # Scene, entities, and camera setup
    │   └── robot_config.py # Robot control parameter configuration
    │
    ├── camera/             # Camera control
    │   └── wrist_camera.py # Wrist-mounted camera pose updates
    │
    └── motion/             # Motion execution
        ├── trajectory.py   # Trajectory execution with logging
        ├── steps.py        # Step-by-step execution
        └── demo.py         # Pick-and-place demonstration
```

## Quick Start

### Prerequisites

- Python 3.8+
- Genesis physics engine
- Required packages: numpy, opencv-python, genesis-world

### Installation

```bash
# Activate virtual environment
source genesis_venv/bin/activate

# Install dependencies (if not already installed)
pip install genesis-world numpy opencv-python
```

### Running the Simulation

```bash
# Run with full visualization (viewer + camera playback)
python base_launch.py

# Run with viewer only (no camera playback)
python base_launch.py --sim-only

# Run headless (no visualization)
python base_launch.py --headless
```

## Data Collection

The simulation automatically collects:

### Proprioceptive Data
- **Joint states**: positions, velocities, torques (9 DOF)
- **End-effector pose**: position and quaternion orientation
- **End-effector velocity**: linear velocity
- **Gripper state**: finger positions and width

### Contact Information
- **Contact detection**: Boolean contact state
- **Contact forces**: Gripper force magnitude
- **Object distance**: Distance between gripper and object
- **Lift detection**: Whether object is lifted off ground

### Object State
- **Object pose**: Position and orientation
- **Object velocity**: Linear velocity for slip detection

### Vision (Optional)
- **RGB images**: Color camera feed (640x480)
- **Depth images**: Depth map from wrist camera

All data is saved to `sensor_data.npz` in compressed NumPy format.

## 🔧 Module Overview

### `src.config`
Handles command-line arguments and Genesis initialization.
- `parse_args()`: Parses headless/sim-only flags
- `init_genesis()`: Initializes Genesis with GPU backend

### `src.sensors`
Complete sensor data collection pipeline.
- `SensorDataLogger`: Main data logging class with timestep tracking
- `get_proprioception()`: Reads joint states and end-effector pose
- `detect_contact_with_object()`: Ground-truth contact detection
- `get_object_state()`: Tracks object pose and velocity
- `collect_sensor_data()`: Unified sensor data collection

### `src.scene`
Scene and robot setup utilities.
- `setup_scene()`: Creates simulation scene with viewer options
- `setup_entities()`: Adds plane, cube, and Franka robot
- `setup_camera()`: Adds wrist-mounted RGB-D camera
- `configure_robot()`: Sets PD control gains and force limits

#### PD Controller Tuning

The `configure_robot()` function supports multiple gain presets to control motion behavior:

```python
# Use different gain presets
configure_robot(franka, preset="default")          # Balanced (recommended)
configure_robot(franka, preset="low_overshoot")    # Slow, stable motion
configure_robot(franka, preset="aggressive")       # Fast tracking, more overshoot
configure_robot(franka, preset="original")         # Legacy gains (causes overshoot)
```

**Understanding the gains:**
- **kp (proportional gain)**: Controls stiffness. Higher values = faster response but more overshoot
- **kv (derivative gain)**: Controls damping. Higher values = smoother motion, less overshoot
- **Critical parameter**: `kv/kp ratio` determines damping behavior
  - Ratio < 0.3: Underdamped (overshoots)
  - Ratio 0.4-0.6: Critically damped (optimal response)
  - Ratio > 0.7: Overdamped (slow, stable)

**Available presets:**
| Preset | kp/kv Ratio | Behavior | Use Case |
|--------|------------|----------|----------|
| `default` | 0.25 | Balanced | General pick-and-place |
| `low_overshoot` | 0.35 | Slow, stable | Fragile objects, high precision |
| `aggressive` | 0.17 | Fast, responsive | Quick movements, can overshoot |
| `original` | 0.11 | Very responsive | Legacy, causes significant overshoot |

To adjust gains further, modify the `GAIN_PRESETS` dictionary in `src/scene/robot_config.py`.


### `src.camera`
Camera control and pose updates.
- `update_wrist_camera()`: Updates camera pose to follow end-effector

### `src.motion`
Motion execution with integrated logging.
- `execute_trajectory()`: Executes planned trajectories with logging
- `execute_steps()`: Runs simulation steps with position/force control
- `run_pick_and_place_demo()`: Complete pick-and-place sequence

## Usage Examples

### Basic Simulation

```python
from src.config import parse_args, init_genesis
from src.sensors import SensorDataLogger
from src.scene import setup_scene, setup_entities, configure_robot
from src.motion import run_pick_and_place_demo

# Initialize
args = parse_args()
init_genesis()

# Setup
logger = SensorDataLogger()
scene = setup_scene(show_viewer=True)
plane, cube, franka = setup_entities(scene)
scene.build()
configure_robot(franka)

# Run demo
motors_dof = np.arange(7)
fingers_dof = np.arange(7, 9)
end_effector = franka.get_link('hand')

run_pick_and_place_demo(franka, scene, cam, end_effector, cube, 
                        logger, motors_dof, fingers_dof)

# Save data
logger.save('sensor_data.npz')
```

### Custom Motion Sequence

```python
from src.motion import execute_steps

# Custom motion with position control
qpos = franka.inverse_kinematics(
    link=end_effector,
    pos=np.array([0.65, 0.0, 0.25]),
    quat=np.array([0, 1, 0, 0])
)

execute_steps(franka, scene, cam, end_effector, cube, logger,
             num_steps=100, motors_dof=motors_dof, qpos=qpos[:-2],
             print_status=True, phase_name="Custom Motion")
```

## 🎓 Research Applications

This framework is designed for:
- **Slip detection research**: Ground-truth contact labeling for training
- **Tactile sensing**: Simulated force/torque data collection
- **Manipulation learning**: Dataset generation for learning-based control
- **Grasp stability**: Contact loss detection and analysis
- **Vision-based manipulation**: RGB-D data for visual servoing

## Output Data Format

The `sensor_data.npz` file contains:

```python
data = np.load('sensor_data.npz')

# Proprioception (N timesteps)
data['q']              # (N, 9) joint positions
data['dq']             # (N, 9) joint velocities
data['tau']            # (N, 9) joint torques
data['ee_pos']         # (N, 3) end-effector position
data['ee_quat']        # (N, 4) end-effector quaternion
data['ee_lin_vel']     # (N, 3) end-effector velocity
data['gripper_width']  # (N,) gripper opening

# Contact info
data['in_contact']     # (N,) boolean contact state
data['contact_force']  # (N,) contact force magnitude
data['cube_lifted']    # (N,) boolean lift state

# Object state
data['obj_pos']        # (N, 3) object position
data['obj_quat']       # (N, 4) object orientation
data['obj_lin_vel']    # (N, 3) object velocity

# Metadata
data['timestep']       # (N,) timestep indices
```

## Extending the Framework

### Adding New Sensors

```python
# In src/sensors/custom_sensor.py
def get_custom_sensor_data(robot, object):
    # Your sensing logic
    return {'custom_data': value}

# In src/sensors/data_collection.py
from .custom_sensor import get_custom_sensor_data

def collect_sensor_data(...):
    # Add to collection
    custom_data = get_custom_sensor_data(franka, cube)
    data.update(custom_data)
```

### Custom Motion Primitives

```python
# In src/motion/custom_motion.py
def custom_motion_primitive(franka, scene, ...):
    # Your motion logic
    pass
```

## License

This project uses the Genesis physics engine. Please refer to Genesis licensing for usage terms.

## Contributing

Contributions are welcome! Please ensure:
- Code follows the modular structure
- Functions include docstrings
- New modules are properly imported in `__init__.py`

## Contact

For questions or issues, please open a GitHub issue.

---

**Note**: This simulation uses ground-truth physics for contact detection. In real-world applications, replace with actual force/torque and tactile sensors.
