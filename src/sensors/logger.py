"""Sensor data logger class for collecting simulation data."""
import numpy as np
from collections import defaultdict
from typing import Dict, List, Any


class SensorDataLogger:
    """
    Collects and stores sensor data for slip detection and dataset creation.
    
    Records:
    - Proprioception: joint states, EE pose/twist, gripper state
    - Contact info: ground-truth contact for auto-labeling
    - Vision: RGBD images (optional)
    """
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Clear all logged data."""
        self.data = defaultdict(list)
        self.timestep = 0
    
    def log_step(self, step_data: Dict[str, Any]):
        """Log data for current timestep."""
        step_data['timestep'] = self.timestep
        for key, value in step_data.items():
            self.data[key].append(value)
        self.timestep += 1
    
    def get_last(self, key: str) -> Any:
        """Get last logged value for a key."""
        if key in self.data and len(self.data[key]) > 0:
            return self.data[key][-1]
        return None
    
    def save(self, filepath: str):
        """Save logged data to npz file."""
        save_dict = {}
        for key, values in self.data.items():
            if isinstance(values[0], np.ndarray):
                save_dict[key] = np.array(values)
            else:
                save_dict[key] = values
        np.savez_compressed(filepath, **save_dict)
        print(f"Saved sensor data to {filepath}")
    
    def get_contact_state_label(self) -> int:
        """
        Get contact state label for current timestep.
        Returns: 0 = no contact, 1 = in contact
        """
        return self.get_last('in_contact') or 0
    
    def detect_contact_lost(self) -> bool:
        """
        Detect if contact was just lost (for event labeling).
        Returns: True if contact lost this timestep
        """
        if len(self.data['in_contact']) < 2:
            return False
        prev_contact = self.data['in_contact'][-2]
        curr_contact = self.data['in_contact'][-1]
        return prev_contact and not curr_contact
    
    def get_summary_stats(self) -> Dict[str, int]:
        """Get summary statistics of collected data."""
        total_timesteps = self.timestep
        total_contact = sum(self.data['in_contact'])
        contact_lost_events = sum([
            self.data['in_contact'][i-1] and not self.data['in_contact'][i] 
            for i in range(1, len(self.data['in_contact']))
        ])
        return {
            'total_timesteps': total_timesteps,
            'total_contact': total_contact,
            'contact_lost_events': contact_lost_events
        }
