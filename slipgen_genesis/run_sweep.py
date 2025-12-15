"""Parameter sweep runner for slip metrics and trajectories."""
import argparse
from slipgen.task_pick_place import run_pick_place_sweep


def parse_args():
    parser = argparse.ArgumentParser(description="Run pick-and-place parameter sweep")
    parser.add_argument("--headless", action="store_true",
                        help="Run without Genesis viewer (no 3D visualization)")
    parser.add_argument("--no-camera", action="store_true",
                        help="Disable OpenCV camera rendering (RGB/depth windows)")
    parser.add_argument("--sim-only", action="store_true",
                        help="Simulation only: equivalent to --headless --no-camera")
    parser.add_argument("--num-cubes", type=int, default=3,
                        help="Number of cubes in scene (default: 3)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    # --sim-only implies both headless and no camera
    show_viewer = not (args.headless or args.sim_only)
    render_cameras = not (args.no_camera or args.sim_only)
    
    run_pick_place_sweep(
        num_cubes=args.num_cubes,
        show_viewer=show_viewer,
        render_cameras=render_cameras,
    )
