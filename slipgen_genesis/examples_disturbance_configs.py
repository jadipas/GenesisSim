"""
Example script demonstrating the new transport disturbance parameters.

This shows how to configure repeatable slip stress conditions for data collection.
"""
import numpy as np

# Example configurations for different disturbance levels

# ============================================================================
# BASELINE: No disturbances (smooth, predictable transport)
# ============================================================================
baseline_config = {
    'transport_steps': 240,
    'phase_warp_level': 0,
    'shake_amp': 0.0,
    'shake_freq': 6.0,
}

# ============================================================================
# MILD: Gentle speedup in middle of transport
# ============================================================================
mild_config = {
    'transport_steps': 240,
    'phase_warp_level': 1,  # 35% phase compression
    'shake_amp': 0.0,
    'shake_freq': 6.0,
}

# ============================================================================
# MODERATE: Medium speedup + small shake
# ============================================================================
moderate_config = {
    'transport_steps': 240,
    'phase_warp_level': 2,  # 65% phase compression
    'shake_amp': 0.015,     # 0.86° amplitude
    'shake_freq': 6.0,      # 6 cycles over trajectory
}

# ============================================================================
# STRONG: Aggressive speedup + larger shake
# ============================================================================
strong_config = {
    'transport_steps': 240,
    'phase_warp_level': 3,  # 100% phase compression
    'shake_amp': 0.025,     # 1.43° amplitude
    'shake_freq': 8.0,      # 8 cycles (higher frequency)
}

# ============================================================================
# FAST: Reduced duration with moderate disturbance
# ============================================================================
fast_config = {
    'transport_steps': 180,  # 25% faster than baseline
    'phase_warp_level': 2,
    'shake_amp': 0.02,
    'shake_freq': 7.0,
}

# ============================================================================
# SLOW: Extended duration for careful manipulation
# ============================================================================
slow_config = {
    'transport_steps': 320,  # 33% slower than baseline
    'phase_warp_level': 1,
    'shake_amp': 0.01,
    'shake_freq': 5.0,
}


# ============================================================================
# Example: Dataset sweep with systematic variation
# ============================================================================
def generate_disturbance_sweep():
    """Generate a systematic sweep of disturbance parameters for dataset collection."""
    configs = []
    
    # Baseline runs (no disturbances, vary speed only)
    for steps in [180, 210, 240, 280, 320]:
        configs.append({
            'name': f'baseline_s{steps}',
            'transport_steps': steps,
            'phase_warp_level': 0,
            'shake_amp': 0.0,
            'shake_freq': 6.0,
        })
    
    # Phase warp sweep (fixed speed, vary warp)
    for warp in [1, 2, 3]:
        configs.append({
            'name': f'warp_l{warp}',
            'transport_steps': 240,
            'phase_warp_level': warp,
            'shake_amp': 0.0,
            'shake_freq': 6.0,
        })
    
    # Shake sweep (fixed speed and warp, vary shake)
    for amp in [0.01, 0.015, 0.02, 0.025, 0.03]:
        for freq in [4.0, 6.0, 8.0]:
            configs.append({
                'name': f'shake_a{int(amp*1000)}_f{int(freq)}',
                'transport_steps': 240,
                'phase_warp_level': 1,
                'shake_amp': amp,
                'shake_freq': freq,
            })
    
    # Combined sweep (warp + shake)
    for warp in [1, 2, 3]:
        for amp in [0.01, 0.02, 0.03]:
            configs.append({
                'name': f'combined_w{warp}_a{int(amp*1000)}',
                'transport_steps': 240,
                'phase_warp_level': warp,
                'shake_amp': amp,
                'shake_freq': 6.0,
            })
    
    return configs


# ============================================================================
# Example usage in a simulation loop
# ============================================================================
def example_usage():
    """
    Example of how to use the new parameters in your simulation loop.
    
    Replace the placeholder comments with your actual setup code.
    """
    
    # Setup (placeholder - replace with your actual setup)
    # franka, scene, cam, end_effector = setup_simulation()
    # cube = spawn_cube(scene)
    # logger = create_logger()
    # motors_dof, fingers_dof = get_dof_indices(franka)
    # drop_pos = np.array([0.55, 0.25, 0.15])
    
    # Choose a configuration
    config = moderate_config  # or baseline_config, strong_config, etc.
    
    # Run pick-and-place with specified disturbances
    # from slipgen.demo import run_pick_and_place_demo
    # run_pick_and_place_demo(
    #     franka=franka,
    #     scene=scene,
    #     cam=cam,
    #     end_effector=end_effector,
    #     cube=cube,
    #     logger=logger,
    #     motors_dof=motors_dof,
    #     fingers_dof=fingers_dof,
    #     display_video=True,
    #     drop_pos=drop_pos,
    #     debug_plot_transfer=True,
    #     **config  # Unpack the configuration
    # )
    
    pass


# ============================================================================
# Example: Random sampling for diverse dataset
# ============================================================================
def random_disturbance_config(seed=None):
    """Generate a random disturbance configuration for dataset diversity."""
    if seed is not None:
        np.random.seed(seed)
    
    return {
        'transport_steps': np.random.choice([180, 210, 240, 280, 320]),
        'phase_warp_level': np.random.choice([0, 1, 2, 3]),
        'shake_amp': np.random.uniform(0.0, 0.03) if np.random.rand() > 0.3 else 0.0,
        'shake_freq': np.random.uniform(4.0, 8.0),
    }


if __name__ == '__main__':
    # Print summary of available configurations
    print("Available Transport Disturbance Configurations:")
    print("=" * 80)
    
    configs = {
        'Baseline': baseline_config,
        'Mild': mild_config,
        'Moderate': moderate_config,
        'Strong': strong_config,
        'Fast': fast_config,
        'Slow': slow_config,
    }
    
    for name, config in configs.items():
        print(f"\n{name}:")
        for key, value in config.items():
            print(f"  {key}: {value}")
    
    # Generate sweep
    print("\n\nDataset Sweep:")
    print("=" * 80)
    sweep_configs = generate_disturbance_sweep()
    print(f"Total configurations: {len(sweep_configs)}")
    print(f"First 5 configurations:")
    for cfg in sweep_configs[:5]:
        print(f"  {cfg['name']}: warp={cfg['phase_warp_level']}, "
              f"shake={cfg['shake_amp']:.3f}, freq={cfg['shake_freq']:.1f}")
    
    # Example random config
    print("\n\nRandom Configuration Example:")
    print("=" * 80)
    for i in range(3):
        cfg = random_disturbance_config(seed=i)
        print(f"Random {i+1}: {cfg}")
