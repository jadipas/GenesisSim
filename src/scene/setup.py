"""Scene and entity setup functions."""
import genesis as gs


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


def setup_entities(scene):
    """Add entities to the scene."""
    plane = scene.add_entity(gs.morphs.Plane())
    cube = scene.add_entity(
        gs.morphs.Box(
            size = (0.04, 0.04, 0.04),
            pos  = (0.65, 0.0, 0.02),
        )
    )
    franka = scene.add_entity(
        gs.morphs.MJCF(file='xml/franka_emika_panda/panda.xml'),
    )
    return plane, cube, franka


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
