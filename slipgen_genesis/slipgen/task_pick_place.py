"""Pick-and-place task API built on migrated demo/steps with slip knobs."""
from typing import List, Optional
import numpy as np

from slipgen.demo import run_iterative_pick_and_place, run_pick_and_place_demo
from slipgen.logger import Logger
from slipgen.knobs import SlipKnobs
from slipgen.scene import setup_with_knobs


def run_pick_place_sweep(num_cubes: int = 3, display_video: bool = True,
                         mu_vals=(0.2, 0.6, 1.0), fn_caps=(2.0, 5.0, 8.0), disturb_levels=(0, 1, 2)):
    """Run parameter sweep with scene reuse for efficiency."""
    from slipgen.scene import init_scene, apply_knobs
    
    # One-time expensive scene initialization
    print("Initializing scene (one-time build)...")
    scene, franka, cam, end_effector, cubes, motors_dof, fingers_dof = init_scene(show_viewer=display_video)
    
    results = []
    total_configs = len(mu_vals) * len(fn_caps) * len(disturb_levels)
    config_idx = 0
    
    for mu in mu_vals:
        for fn in fn_caps:
            for d in disturb_levels:
                config_idx += 1
                print(f"\n[{config_idx}/{total_configs}] Testing config: mu={mu}, fn_cap={fn}, disturb={d}")
                
                knobs = SlipKnobs(mu=mu, fn_cap=fn, disturb_level=d)
                apply_knobs(knobs, scene, franka, cubes, fingers_dof)
                
                logger = Logger()
                # TODO: Pass disturb_level to demo to enable trajectory disturbance
                run_iterative_pick_and_place(franka, scene, cam, end_effector, cubes, logger, motors_dof, fingers_dof, display_video=display_video)
                stats = logger.get_slippage_metrics()
                results.append({'mu': mu, 'fn_cap': fn, 'disturb': d, **stats})
                print(f"  Result: {stats}")
                
                # Save force plot for this configuration
                filename = f"force_plot_sweep_mu{mu}_fn{fn}_d{d}.png"
                logger.save_force_plot(output_dir=".", filename=filename)
    
    print(f"\nSweep finished: tested {total_configs} configurations.")
    return results


def generate_dataset(num_samples: int = 10, display_video: bool = False,
                     mu: float = 0.6, fn_cap: float = 5.0, disturb_level: int = 1,
                     save_path: str = "sensor_data.npz"):
    knobs = SlipKnobs(mu=mu, fn_cap=fn_cap, disturb_level=disturb_level)
    scene, franka, cam, end_effector, cubes, motors_dof, fingers_dof = setup_with_knobs(knobs, show_viewer=display_video)
    logger = Logger()
    for i in range(num_samples):
        cube = cubes[i % len(cubes)]
        drop_pos = np.array([0.55, 0.38, 0.14])
        run_pick_and_place_demo(franka, scene, cam, end_effector, cube, logger, motors_dof, fingers_dof,
                                display_video=display_video, drop_pos=drop_pos)
    logger.save(save_path)
    
    # Save final force plot for entire dataset
    logger.save_force_plot(output_dir=".", filename="dataset_force_plot.png")
