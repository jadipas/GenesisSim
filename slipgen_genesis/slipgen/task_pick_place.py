"""Pick-and-place task API built on migrated demo/steps with slip knobs."""
from typing import List, Optional
import numpy as np

from slipgen.demo import run_iterative_pick_and_place, run_pick_and_place_demo
from slipgen.logger import Logger
from slipgen.knobs import SlipKnobs
from slipgen.scene import setup_with_knobs, reset_cube_positions


def run_pick_place_sweep(num_cubes: int = 3, show_viewer: bool = True, render_cameras: bool = True,
                         mu_vals=(0.2, 0.6, 1.0), q_targets=(-0.02, -0.03, -0.04), disturb_levels=(0, 1, 2)):
    """Run parameter sweep with scene reuse for efficiency.
    
    Args:
        num_cubes: Number of cubes in the scene
        show_viewer: Show Genesis 3D viewer window
        render_cameras: Render OpenCV RGB/depth camera windows
        mu_vals: Friction coefficient values to sweep
        fn_caps: Force cap values to sweep
        disturb_levels: Disturbance levels to sweep
    """
    from slipgen.scene import init_scene, apply_knobs
    
    # One-time expensive scene initialization
    print("Initializing scene (one-time build)...")
    print(f"  show_viewer={show_viewer}, render_cameras={render_cameras}")
    cube_area = {
        "x_range": (0.35, 0.55),
        "y_range": (-0.55, -0.05),
        "z": 0.035,
        "min_separation": 0.08,
    }
    scene, franka, cam, end_effector, cubes, motors_dof, fingers_dof = init_scene(show_viewer=show_viewer, cube_area=cube_area)
    results = []
    total_configs = len(mu_vals) * len(q_targets) * len(disturb_levels)
    config_idx = 0
    
    for mu in mu_vals:
        for q in q_targets:
            for d in disturb_levels:
                config_idx += 1
                print(f"\n[{config_idx}/{total_configs}] Testing config: mu={mu}, q_finger_target={q}, disturb={d}")
                
                knobs = SlipKnobs(mu=mu, disturb_level=d, q_finger_target=q)
                apply_knobs(knobs, scene, franka, cubes, fingers_dof)
                
                logger = Logger()
                run_iterative_pick_and_place(franka, scene, cam, end_effector, cubes, logger, motors_dof, fingers_dof, 
                                            show_viewer=show_viewer, render_cameras=render_cameras, knobs=knobs)
                stats = logger.get_slippage_metrics()
                results.append({'mu': mu, 'q_finger_target': q, 'disturb': d, **stats})
                print(f"  Result: {stats}")
                
                # Save force plot for this configuration
                filename = f"force_plot_sweep_mu{mu}_q{q}_d{d}.png"
                # logger.save_force_plot(output_dir=".", filename=filename)
                reset_cube_positions(cubes, cube_area)
    
    print(f"\nSweep finished: tested {total_configs} configurations.")
    return results


def generate_dataset(num_samples: int = 10, show_viewer: bool = False, render_cameras: bool = False,
                     mu: float = 0.6, fn_cap: float = 5.0, disturb_level: int = 1,
                     save_path: str = "sensor_data.npz"):
    """Generate dataset from pick-and-place runs.
    
    Args:
        num_samples: Number of pick-and-place cycles
        show_viewer: Show Genesis 3D viewer window
        render_cameras: Render OpenCV RGB/depth camera windows
        mu: Friction coefficient
        fn_cap: Force cap
        disturb_level: Disturbance level
        save_path: Path to save the sensor data
    """
    knobs = SlipKnobs(mu=mu, fn_cap=fn_cap, disturb_level=disturb_level)
    scene, franka, cam, end_effector, cubes, motors_dof, fingers_dof = setup_with_knobs(knobs, show_viewer=show_viewer)
    logger = Logger()
    for i in range(num_samples):
        cube = cubes[i % len(cubes)]
        drop_pos = np.array([0.55, 0.38, 0.14])
        run_pick_and_place_demo(franka, scene, cam, end_effector, cube, logger, motors_dof, fingers_dof,
                                show_viewer=show_viewer, render_cameras=render_cameras, drop_pos=drop_pos, knobs=knobs)
    logger.save(save_path)
    
    # Save final force plot for entire dataset
    # logger.save_force_plot(output_dir=".", filename="dataset_force_plot.png")
