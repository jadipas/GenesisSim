"""Slipgen-facing logger extending SensorDataLogger with metadata and slip metrics."""
from typing import Dict, Any
from collections import defaultdict
import numpy as np
from slipgen.force_viz import ForceVisualizer


class Logger:
    """Collects and stores sensor data for slip detection and dataset creation."""
    
    def __init__(self):
        self.visualizer = ForceVisualizer(title="Gripper Force Monitoring")
        self.reset()
    
    def reset(self):
        """Clear all logged data."""
        self.data = defaultdict(list)
        self.timestep = 0
        self.phase_markers = {}
        self.grasp_phase_contact = []
        self.current_phase: str | None = None
        self.cycle_count: int = 0
        # Keep visualizer alive across resets

    def log_step(self, step_data: Dict[str, Any]):
        """Log data for current timestep."""
        step_data['timestep'] = self.timestep
        for key, value in step_data.items():
            self.data[key].append(value)
        
        # Collect force data for later plotting
        left_force = step_data.get('left_finger_force', 0.0)
        right_force = step_data.get('right_finger_force', 0.0)
        self.visualizer.add_measurement(left_force, right_force, self.timestep)
        
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
        """Get contact state label for current timestep."""
        return self.get_last('in_contact') or 0
    
    def detect_contact_lost(self) -> bool:
        """Detect if contact was just lost."""
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
        
        slippage_occurred = False
        if 'Grasping' in self.phase_markers and 'Releasing' in self.phase_markers:
            grasp_end = self.phase_markers['Grasping']['end']
            release_start = self.phase_markers['Releasing']['start']
            
            if grasp_end < release_start and grasp_end < len(self.data['in_contact']):
                contact_during_transport = self.data['in_contact'][grasp_end:release_start]
                slippage_occurred = not all(contact_during_transport)
        
        return {
            'total_timesteps': total_timesteps,
            'total_contact': total_contact,
            'contact_lost_events': contact_lost_events,
            'slippage_occurred': slippage_occurred,
            'slippage_count': contact_lost_events,
        }
    
    def mark_phase_start(self, phase_name: str):
        """Mark the start of a phase."""
        if phase_name not in self.phase_markers:
            self.phase_markers[phase_name] = {}
        self.phase_markers[phase_name]['start'] = self.timestep
        self.current_phase = phase_name
        
        # Mark phase in visualizer for later plotting
        phase_label = f"{phase_name} (C{self.cycle_count})" if self.cycle_count > 0 else phase_name
        self.visualizer.mark_phase(phase_label, self.timestep)
    
    def mark_phase_end(self, phase_name: str, show_graph: bool = False):
        """Mark the end of a phase."""
        if phase_name not in self.phase_markers:
            self.phase_markers[phase_name] = {}
        self.phase_markers[phase_name]['end'] = self.timestep
        
        if phase_name == 'Grasping':
            phase_start = self.phase_markers[phase_name].get('start', 0)
            phase_end = self.phase_markers[phase_name].get('end', self.timestep)
            self.grasp_phase_contact = self.data['in_contact'][phase_start:phase_end]
    
    def get_slippage_metrics(self) -> Dict[str, Any]:
        """Get detailed slippage metrics for the pick-and-place task."""
        metrics = {
            'slippage_occurred': False,
            'grasp_to_drop_contact_loss': 0,
            'transport_phase_contact_pct': 100.0,
            'grasp_phase_contact_pct': 100.0,
        }
        
        if 'Grasping' in self.phase_markers and 'Releasing' in self.phase_markers:
            grasp_start = self.phase_markers['Grasping'].get('start', 0)
            grasp_end = self.phase_markers['Grasping'].get('end', self.timestep)
            release_start = self.phase_markers['Releasing'].get('start', self.timestep)
            
            if grasp_start < grasp_end:
                grasp_contact = self.data['in_contact'][grasp_start:grasp_end]
                grasp_contact_pct = (sum(grasp_contact) / len(grasp_contact) * 100) if grasp_contact else 0
                metrics['grasp_phase_contact_pct'] = grasp_contact_pct
            
            if grasp_end < release_start:
                transport_contact = self.data['in_contact'][grasp_end:release_start]
                if transport_contact:
                    transport_contact_pct = sum(transport_contact) / len(transport_contact) * 100
                    metrics['transport_phase_contact_pct'] = transport_contact_pct
                    metrics['slippage_occurred'] = transport_contact_pct < 95.0
                    
                    loss_events = sum([
                        not transport_contact[i] and transport_contact[i-1]
                        for i in range(1, len(transport_contact))
                    ])
                    metrics['grasp_to_drop_contact_loss'] = loss_events
        
        return metrics
    
    def save_force_plot(self, output_dir: str = ".", filename: str = "force_plot.png"):
        """Generate and save force plot from collected data."""
        self.visualizer.generate_and_save_plot(output_dir, filename)
    
    def reset_visualizer(self):
        """Reset visualizer for a new experiment."""
        self.visualizer = ForceVisualizer(title="Gripper Force Monitoring")
