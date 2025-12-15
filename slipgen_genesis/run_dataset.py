"""Dataset generation entrypoint."""
import argparse
from slipgen.task_pick_place import generate_dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Generate pick-and-place dataset")
    parser.add_argument("--headless", action="store_true",
                        help="Run without Genesis viewer (no 3D visualization)")
    parser.add_argument("--no-camera", action="store_true",
                        help="Disable OpenCV camera rendering (RGB/depth windows)")
    parser.add_argument("--sim-only", action="store_true",
                        help="Simulation only: equivalent to --headless --no-camera")
    parser.add_argument("--num-samples", type=int, default=10,
                        help="Number of pick-and-place cycles (default: 10)")
    parser.add_argument("--mu", type=float, default=0.6,
                        help="Friction coefficient (default: 0.6)")
    parser.add_argument("--fn-cap", type=float, default=5.0,
                        help="Force cap in Newtons (default: 5.0)")
    parser.add_argument("--disturb-level", type=int, default=1,
                        help="Disturbance level 0-3 (default: 1)")
    parser.add_argument("--output", type=str, default="sensor_data.npz",
                        help="Output file path (default: sensor_data.npz)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    # --sim-only implies both headless and no camera
    show_viewer = not (args.headless or args.sim_only)
    render_cameras = not (args.no_camera or args.sim_only)
    
    generate_dataset(
        num_samples=args.num_samples,
        show_viewer=show_viewer,
        render_cameras=render_cameras,
        mu=args.mu,
        fn_cap=args.fn_cap,
        disturb_level=args.disturb_level,
        save_path=args.output,
    )
