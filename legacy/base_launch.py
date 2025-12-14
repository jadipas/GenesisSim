"""Genesis Franka Pick-and-Place Simulation - Main Launch Script."""
import numpy as np
import time
import os

# Suppress Genesis verbose output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Import configuration
from src.config import parse_args, init_genesis

# Import sensors
from src.sensors import SensorDataLogger

# Import scene setup
from src.scene import setup_scene, setup_entities, setup_camera, configure_robot

# Import camera control
from src.camera import update_wrist_camera

# Import motion execution
from src.motion import run_iterative_pick_and_place

# Parse arguments and initialize
args = parse_args()
init_genesis()

def main():
    """Main execution function."""
    # Determine graphics mode
    show_viewer = not args.headless
    show_camera_playback = not (args.headless or args.sim_only)
    
    # Initialize logger
    logger = SensorDataLogger()
    
    # Setup scene
    scene = setup_scene(show_viewer=show_viewer)
    num_cubes = np.random.randint(3, 6)
    plane, cubes, franka = setup_entities(scene, num_cubes=num_cubes)
    cam = setup_camera(scene) if show_camera_playback else None
    
    # Build scene
    scene.build()
    
    # Configure robot
    motors_dof = np.arange(7)
    fingers_dof = np.arange(7, 9)
    configure_robot(franka)
    
    time.sleep(1.0)
    
    end_effector = franka.get_link('hand')
    if cam is not None:
        update_wrist_camera(cam, end_effector)
    
    # Run iterative multi-cube demonstration
    run_iterative_pick_and_place(
        franka,
        scene,
        cam,
        end_effector,
        cubes,
        logger,
        motors_dof,
        fingers_dof,
        display_video=show_camera_playback,
    )
    
    # Save and report results
    logger.save('sensor_data.npz')
    stats = logger.get_summary_stats()
    slippage_metrics = logger.get_slippage_metrics()
    
    print(f"\nCollected {stats['total_timesteps']} timesteps of sensor data")
    print(f"Total contact events: {stats['total_contact']}")
    print(f"Contact lost events: {stats['contact_lost_events']}")
    print(f"\n--- Slippage Analysis ---")
    print(f"Slippage occurred: {slippage_metrics['slippage_occurred']}")
    print(f"Contact loss events during transport: {slippage_metrics['grasp_to_drop_contact_loss']}")
    print(f"Grasp phase contact: {slippage_metrics['grasp_phase_contact_pct']:.1f}%")
    print(f"Transport phase contact: {slippage_metrics['transport_phase_contact_pct']:.1f}%")


if __name__ == "__main__":
    main()

