"""Dataset generation for pick-and-hold trials."""
import argparse
import numpy as np

from slipgen.demo import run_pick_and_hold_demo
from slipgen.logger import Logger
from slipgen.knobs import SlipKnobs
from slipgen.scene import setup_with_knobs, reset_cube_positions, scatter_cubes
from slipgen.task_pick_place import ProgressBarWindow


def parse_args():
    parser = argparse.ArgumentParser(description="Generate pick-and-hold dataset")
    parser.add_argument("--headless", action="store_true",
                        help="Run without Genesis viewer (no 3D visualization)")
    parser.add_argument("--no-camera", action="store_true",
                        help="Disable OpenCV camera rendering (RGB/depth windows)")
    parser.add_argument("--sim-only", action="store_true",
                        help="Simulation only: equivalent to --headless --no-camera")
    parser.add_argument("--num-samples", type=int, default=10,
                        help="Number of pick-and-hold trials (default: 10)")
    parser.add_argument("--mu", type=float, default=0.6,
                        help="Friction coefficient (default: 0.6)")
    parser.add_argument("--fn-cap", type=float, default=5.0,
                        help="Force cap in Newtons (default: 5.0)")
    parser.add_argument("--q-finger-target", type=float, default=-0.03,
                        help="Finger position target (default: -0.03)")
    parser.add_argument("--lift-height", type=float, default=0.1,
                        help="Lift height in meters (default: 0.1)")
    parser.add_argument("--output", type=str, default="hold_sensor_data.npz",
                        help="Output file path (default: hold_sensor_data.npz)")
    return parser.parse_args()


def generate_hold_dataset(
    num_samples: int = 10,
    show_viewer: bool = False,
    render_cameras: bool = False,
    mu: float = 0.6,
    fn_cap: float = 5.0,
    q_finger_target: float = -0.03,
    lift_height: float = 0.1,
    save_path: str = "hold_sensor_data.npz",
):
    """Generate dataset from pick-and-hold runs.
    
    Args:
        num_samples: Number of pick-and-hold trials
        show_viewer: Show Genesis 3D viewer window
        render_cameras: Render OpenCV RGB/depth camera windows
        mu: Friction coefficient (consistent across all trials)
        fn_cap: Force cap in Newtons (consistent across all trials)
        q_finger_target: Finger position target
        lift_height: Height to lift the object
        save_path: Path to save the sensor data
    """
    # Create knobs with consistent mu and force cap
    knobs = SlipKnobs(mu=mu, fn_cap=fn_cap, q_finger_target=q_finger_target, disturb_level=0)
    
    cube_area = {
        "x_range": (0.25, 0.45),
        "y_range": (-0.35, -0.35),
        "z": 0.035,
        "min_separation": 0.08,
    }
    
    # Initialize scene with a single cube (pick-and-hold works with one object at a time)
    scene, franka, cam, end_effector, cubes, motors_dof, fingers_dof = setup_with_knobs(
        knobs, show_viewer=show_viewer, cube_area=cube_area, num_cubes=1
    )
    
    logger = Logger()
    
    # Launch progress bar in a separate terminal window
    pbar = ProgressBarWindow(total=num_samples, desc="Generating hold samples")
    
    for i in range(num_samples):
        # Sample grasp angle uniformly between [0, 90] degrees
        grasp_tilt_angle = np.random.uniform(0.0, 90.0)
        
        print(f"\n[Trial {i+1}/{num_samples}] grasp_tilt_angle={grasp_tilt_angle:.2f}°, mu={mu}, fn_cap={fn_cap}")
        
        # Reset cube position for this trial
        reset_cube_positions(cubes, cube_area)
        
        # Run pick-and-hold demo with sampled grasp angle
        cube = cubes[0]
        run_pick_and_hold_demo(
            franka=franka,
            scene=scene,
            cam=cam,
            end_effector=end_effector,
            cube=cube,
            logger=logger,
            motors_dof=motors_dof,
            fingers_dof=fingers_dof,
            render_cameras=render_cameras,
            lift_height=lift_height,
            debug_plot_transfer=False,
            knobs=knobs,
            grasp_tilt_angle=grasp_tilt_angle,
        )
        
        pbar.update(1)
    
    pbar.close()
    logger.save(save_path)
    print(f"\nDataset saved to {save_path} ({num_samples} samples)")


if __name__ == "__main__":
    args = parse_args()
    
    # --sim-only implies both headless and no camera
    show_viewer = not (args.headless or args.sim_only)
    render_cameras = not (args.no_camera or args.sim_only)
    
    generate_hold_dataset(
        num_samples=args.num_samples,
        show_viewer=show_viewer,
        render_cameras=render_cameras,
        mu=args.mu,
        fn_cap=args.fn_cap,
        q_finger_target=args.q_finger_target,
        lift_height=args.lift_height,
        save_path=args.output,
    )
