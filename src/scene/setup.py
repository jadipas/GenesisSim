"""Scene and entity setup functions."""
import genesis as gs
import numpy as np


def setup_scene(show_viewer: bool = True):
    """Create and configure the simulation scene.
    
    Args:
        show_viewer: Whether to display the viewer
    """
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


def setup_entities(scene, num_cubes: int = 1, cube_positions=None, cube_area=None):
    """Add entities to the scene with configurable cube spawning."""
    plane = scene.add_entity(gs.morphs.Plane())

    # Define spawn area in front of the robot
    cube_area = cube_area or {
        "x_range": (0.50, 0.82),
        "y_range": (-0.20, 0.20),
        "z": 0.02,
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
                size=(0.04, 0.04, 0.04),
                pos=tuple(pos),
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
