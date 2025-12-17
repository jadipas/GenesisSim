"""Test script for run_pick_and_hold_demo - moves robot to pre-grasp position."""
import numpy as np
from slipgen.scene import init_scene, apply_knobs
from slipgen.knobs import SlipKnobs
from slipgen.logger import Logger
from slipgen.demo import run_pick_and_hold_demo


def main():
    # Configuration
    show_viewer = True
    render_cameras = False
    grasp_tilt_angle = 45.0  # 90=parallel to floor, 0=perpendicular
    
    # Create knobs with default friction and force settings
    knobs = SlipKnobs(
        mu=0.6,
        fn_cap=10.0,
        q_finger_target=-0.04,
    )
    
    # Define cube spawn area (single cube in reachable area)
    cube_area = {
        "x_range": (0.45, 0.50),
        "y_range": (-0.15, -0.10),
        "z": 0.035,
        "min_separation": 0.08,
    }
    
    print("[TEST] Initializing scene...")
    scene, franka, cam, end_effector, cubes, motors_dof, fingers_dof = init_scene(
        show_viewer=show_viewer,
        cube_area=cube_area,
        num_cubes=1,
    )
    
    # Apply knobs (friction, force limits)
    apply_knobs(knobs, scene, franka, cubes, fingers_dof)
    
    # Create logger for data collection
    logger = Logger()
    
    # Get the single cube
    cube = cubes[0]
    
    print(f"[TEST] Cube position: {cube.get_pos()}")
    print(f"[TEST] Cube quaternion: {cube.get_quat()}")
    print(f"[TEST] Grasp tilt angle: {grasp_tilt_angle}°")
    
    # Run pick-and-hold demo
    print("[TEST] Running pick_and_hold_demo...")
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
        lift_height=0.3,
        grasp_tilt_angle=grasp_tilt_angle,
        knobs=knobs,
    )
    
    print("[TEST] Done!")
    
    # Keep viewer open
    if show_viewer:
        print("[TEST] Press Ctrl+C to exit...")
        try:
            while True:
                scene.step()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
