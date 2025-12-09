"""Robot configuration functions."""
import numpy as np


# Preset PD gain configurations for different control behaviors
GAIN_PRESETS = {
    "default": {
        "kp": np.array([2500, 2500, 2000, 2000, 1500, 1500, 1500, 80, 80]),
        "kv": np.array([650, 650, 600, 600, 400, 400, 400, 20, 20]),
    },
    "low_overshoot": {
        # Higher damping relative to stiffness for slower, more stable motion
        # kv/kp ratio ~0.7 provides overdamped response
        "kp": np.array([1800, 1800, 1500, 1500, 1000, 1000, 1000, 60, 60]),
        "kv": np.array([700, 700, 650, 650, 450, 450, 450, 25, 25]),
    },
    "moderate": {
        # Balanced response: kv/kp ratio ~0.5
        "kp": np.array([2500, 2500, 2000, 2000, 1500, 1500, 1500, 80, 80]),
        "kv": np.array([650, 650, 600, 600, 400, 400, 400, 20, 20]),
    },
    "aggressive": {
        # Lower damping for faster tracking but higher overshoot
        # kv/kp ratio ~0.3
        "kp": np.array([3500, 3500, 2800, 2800, 2000, 2000, 2000, 100, 100]),
        "kv": np.array([550, 550, 450, 450, 300, 300, 300, 15, 15]),
    },
    "original": {
        # Original values (causes overshoot)
        "kp": np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100]),
        "kv": np.array([450, 450, 350, 350, 200, 200, 200, 10, 10]),
    },
}


def configure_robot(franka, preset="default"):
    """
    Set control gains and force limits for the robot.
    
    Args:
        franka: Robot entity
        preset: Gain preset name. Options: "default", "low_overshoot", "moderate", "aggressive", "original"
    """
    if preset not in GAIN_PRESETS:
        raise ValueError(f"Unknown preset: {preset}. Available: {list(GAIN_PRESETS.keys())}")
    
    gains = GAIN_PRESETS[preset]
    franka.set_dofs_kp(gains["kp"])
    franka.set_dofs_kv(gains["kv"])
    
    print(f"[Robot Config] Using '{preset}' gains (kp/kv ratio: {np.mean(gains['kp'] / gains['kv']):.2f})")
    
    franka.set_dofs_force_range(
        np.array([-87, -87, -87, -87, -12, -12, -12, -100, -100]),
        np.array([ 87,  87,  87,  87,  12,  12,  12,  100,  100]),
    )
