"""Force visualization for debugging contact."""
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for performance
import matplotlib.pyplot as plt
import os
from typing import List, Dict


class ForceVisualizer:
    """Collects force data and generates plot at the end."""
    
    def __init__(self, title: str = "Gripper Force Monitoring"):
        self.title = title
        self.left_forces: List[float] = []
        self.right_forces: List[float] = []
        self.timestamps: List[int] = []
        self.phase_markers: Dict[str, int] = {}  # Track phase transitions
        self.phase_names: List[str] = []  # Order of phases
    
    def add_measurement(self, left_force: float, right_force: float, timestep: int):
        """Record force measurement for later plotting."""
        self.left_forces.append(left_force)
        self.right_forces.append(right_force)
        self.timestamps.append(timestep)
    
    def mark_phase(self, phase_name: str, timestep: int):
        """Mark the start of a new phase in the visualization."""
        if phase_name not in self.phase_markers:
            self.phase_markers[phase_name] = timestep
            self.phase_names.append(phase_name)
    
    def generate_and_save_plot(self, output_dir: str = ".", filename: str = "force_plot.png"):
        """Generate and save plot from collected data."""
        if len(self.timestamps) == 0:
            print("[ForceViz] No data to plot")
            return
        
        output_path = os.path.join(output_dir, filename)
        
        # Create a new static figure
        fig, ax = plt.subplots(figsize=(14, 7))
        ax.plot(self.timestamps, self.left_forces, 'b-', linewidth=2, label='Left Finger', marker='o', markersize=2)
        ax.plot(self.timestamps, self.right_forces, 'r-', linewidth=2, label='Right Finger', marker='s', markersize=2)
        ax.axhline(y=0.1, color='gray', linestyle='--', alpha=0.5, label='Contact Threshold (0.1N)')
        
        # Draw phase markers
        for phase_name in self.phase_names:
            if phase_name in self.phase_markers:
                ts = self.phase_markers[phase_name]
                ax.axvline(x=ts, color='green', linestyle=':', alpha=0.7, linewidth=1.5)
                y_pos = ax.get_ylim()[1] * 0.95
                ax.text(ts, y_pos, phase_name, rotation=90, 
                       verticalalignment='top', fontsize=9, alpha=0.8)
        
        ax.set_xlabel('Timestep', fontsize=12)
        ax.set_ylabel('Force (N)', fontsize=12)
        ax.set_title(self.title, fontsize=14, fontweight='bold')
        ax.legend(loc='upper left', fontsize=11)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_path, dpi=100, bbox_inches='tight')
        plt.close(fig)
        
        print(f"[ForceViz] Saved plot to {output_path}")
    
    def summary(self) -> dict:
        if len(self.left_forces) == 0:
            return {
                'left_min': 0, 'left_max': 0, 'left_mean': 0,
                'right_min': 0, 'right_max': 0, 'right_mean': 0,
                'duration': 0
            }
        
        return {
            'left_min': float(np.min(self.left_forces)),
            'left_max': float(np.max(self.left_forces)),
            'left_mean': float(np.mean(self.left_forces)),
            'right_min': float(np.min(self.right_forces)),
            'right_max': float(np.max(self.right_forces)),
            'right_mean': float(np.mean(self.right_forces)),
            'duration': len(self.timestamps),
        }
