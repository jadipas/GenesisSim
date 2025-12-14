"""Scene + entity setup with slip knobs wiring."""
from typing import Tuple, List
import numpy as np
import time
import genesis as gs

from slipgen.knobs import SlipKnobs
from slipgen.camera import mount_wrist_camera

# Preset PD gain configurations
GAIN_PRESETS = {
    "default": {
        "kp": np.array([2500, 2500, 2000, 2000, 1500, 1500, 1500, 80, 80]),
        "kv": np.array([650, 650, 600, 600, 400, 400, 400, 20, 20]),
    },
}


def setup_scene(show_viewer: bool = True):
    """Create and configure the simulation scene."""
    gs.init()
    scene = gs.Scene(
        sim_options = gs.options.SimOptions(dt = 0.01),
        viewer_options = gs.options.ViewerOptions(
            camera_pos    = (3, -1, 1.5),
            camera_lookat = (0.0, 0.0, 0.5),
            camera_fov    = 30,
            max_FPS       = 60,
        ),
        show_viewer = show_viewer,
    )
    return scene


def _sample_cube_positions(num_cubes: int, x_range, y_range, z: float, min_separation: float):
    positions = []
    max_trials = max(50, num_cubes * 30)
    trials = 0
    while len(positions) < num_cubes and trials < max_trials:
        candidate = np.array([
            np.random.uniform(*x_range),
            np.random.uniform(*y_range),
            z,
        ])
        if all(np.linalg.norm(candidate[:2] - np.array(p)[:2]) >= min_separation for p in positions):
            positions.append(candidate)
        trials += 1
    while len(positions) < num_cubes:
        positions.append(np.array([
            np.random.uniform(*x_range),
            np.random.uniform(*y_range),
            z,
        ]))
    return positions


def _random_quaternion():
    """Generate a random unit quaternion for rotation around z-axis only."""
    angle = np.random.uniform(0, 2 * np.pi)
    return np.array([
        np.cos(angle / 2),
        0.0,
        0.0,
        np.sin(angle / 2),
    ])


def setup_entities(scene, num_cubes: int = 1, cube_positions=None, cube_area=None):
    """Add entities to the scene with configurable cube spawning."""
    plane = scene.add_entity(gs.morphs.Plane())

    cube_area = cube_area or {
        "x_range": (0.35, 0.55),
        "y_range": (-0.55, -0.05),
        "z": 0.035,
        "min_separation": 0.08,
    }

    if cube_positions is None:
        cube_positions = _sample_cube_positions(
            num_cubes,
            cube_area["x_range"],
            cube_area["y_range"],
            cube_area["z"],
            cube_area["min_separation"],
        )

    cubes = [
        scene.add_entity(
            gs.morphs.Box(
                size=(0.05, 0.05, 0.05),
                pos=tuple(pos),
                quat=tuple(_random_quaternion()),
            )
        )
        for pos in cube_positions
    ]

    franka = scene.add_entity(
        gs.morphs.MJCF(file='xml/franka_emika_panda/panda.xml'),
    )
    return plane, cubes, franka


def setup_camera(scene):
    """Add wrist-mounted camera to the scene."""
    cam = scene.add_camera(
        res    = (640, 480),
        pos    = (0.6, 0.0, 0.3),
        lookat = (0.7, 0.0, 0.3),
        fov    = 60,
        GUI    = False,
    )
    return cam


def _guess_dofs(franka) -> Tuple[List[int], List[int]]:
    """Heuristic: arm motors are first 7, fingers are last 2."""
    try:
        n = int(franka.get_num_dofs())
    except Exception:
        # Fallback: try shape of qpos
        try:
            q = franka.get_qpos()
            if hasattr(q, 'shape'):
                n = q.shape[0] if len(q.shape) > 0 else 9
            else:
                n = len(q) if hasattr(q, '__len__') else 9
        except Exception:
            n = 9  # Default Franka has 7 arm + 2 finger DOFs
    
    motors = list(range(7)) if n >= 7 else list(range(max(0, n - 2)))
    fingers = [n - 2, n - 1] if n >= 2 else []
    return motors, fingers


def configure_robot(franka, preset="default"):
    """Set control gains and force limits for the robot."""
    if preset not in GAIN_PRESETS:
        raise ValueError(f"Unknown preset: {preset}. Available: {list(GAIN_PRESETS.keys())}")
    
    gains = GAIN_PRESETS[preset]
    franka.set_dofs_kp(gains["kp"])
    franka.set_dofs_kv(gains["kv"])
    
    print(f"[Robot Config] Using '{preset}' gains (kp/kv ratio: {np.mean(gains['kp'] / gains['kv']):.2f})")
    
    # Set force ranges for all DOFs (arm + fingers)
    franka.set_dofs_force_range(
        np.array([-87, -87, -87, -87, -12, -12, -12, -100, -100]),
        np.array([ 87,  87,  87,  87,  12,  12,  12,  100,  100]),
    )


def init_scene(show_viewer: bool = True):
    """One-time scene initialization: create scene, spawn entities, build once."""
    scene = setup_scene(show_viewer=show_viewer)
    plane, cubes, franka = setup_entities(scene, num_cubes=3)
    cam = setup_camera(scene)
    
    # Build the scene (expensive operation - only once)
    scene.build()
    
    # Configure robot PD gains and force limits (CRITICAL for control)
    motors_dof = np.arange(7)
    fingers_dof = np.arange(7, 9)
    configure_robot(franka)
    
    # Let robot settle to initial position
    time.sleep(1.0)

    # End-effector link: use correct link name
    try:
        end_effector = franka.get_link('hand')
    except Exception:
        try:
            end_effector = franka.find_link('hand')
        except Exception:
            end_effector = franka

    # Mount camera to the hand link
    mount_wrist_camera(cam, end_effector)

    return scene, franka, cam, end_effector, cubes, motors_dof, fingers_dof


def apply_knobs(knobs: SlipKnobs, scene, franka, cubes, fingers_dof):
    """Apply slip knobs to existing scene (fast, repeated per config)."""
    # Apply friction on cubes (Knob A)
    for cube in cubes:
        try:
            mat = cube.get_material()
            if hasattr(mat, 'set_friction'):
                mat.set_friction(knobs.mu)
            else:
                cube.set_friction(knobs.mu)
        except Exception:
            pass

    # Update finger force cap (Knob B) - override default force range for fingers only
    try:
        # Set finger force limits while preserving arm limits
        franka.set_dofs_force_range(
            np.array([-87, -87, -87, -87, -12, -12, -12, -knobs.fn_cap, -knobs.fn_cap]),
            np.array([ 87,  87,  87,  87,  12,  12,  12,  knobs.fn_cap,  knobs.fn_cap]),
        )
    except Exception:
        pass


def setup_with_knobs(knobs: SlipKnobs, show_viewer: bool = True):
    """Legacy combined setup (creates scene + applies knobs in one call)."""
    scene, franka, cam, end_effector, cubes, motors_dof, fingers_dof = init_scene(show_viewer=show_viewer)
    apply_knobs(knobs, scene, franka, cubes, fingers_dof)
    return scene, franka, cam, end_effector, cubes, motors_dof, fingers_dof
