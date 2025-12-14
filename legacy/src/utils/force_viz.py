"""Real-time force visualization for debugging contact."""
import numpy as np
import matplotlib
# Use Agg backend (non-interactive) for headless environments
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import subprocess
import os
from typing import List, Tuple


class ForceVisualizer:
    """Live plot of gripper forces during manipulation."""
    
    def __init__(self, phase_name: str = ""):
        """Initialize force visualizer.
        
        Args:
            phase_name: Name of the phase (e.g., "Grasping", "Transport")
        """
        self.phase_name = phase_name
        self.left_forces: List[float] = []
        self.right_forces: List[float] = []
        self.timestamps: List[int] = []
        self.fig = None
        self.ax = None
    
    def add_measurement(self, left_force: float, right_force: float, timestep: int):
        """Record a force measurement.
        
        Args:
            left_force: Left finger force (N)
            right_force: Right finger force (N)
            timestep: Current timestep
        """
        self.left_forces.append(left_force)
        self.right_forces.append(right_force)
        self.timestamps.append(timestep)
    
    def plot(self, block: bool = True, output_dir: str = "/tmp"):
        """Display the force graph.
        
        Args:
            block: If True, pause execution until user closes window
            output_dir: Directory to save plot images
        """
        if len(self.timestamps) == 0:
            print("[ForceViz] No data to plot")
            return
        
        # Create figure
        plt.figure(figsize=(12, 6))
        
        # Plot forces
        plt.plot(self.timestamps, self.left_forces, 'b-', linewidth=2, label='Left Finger', marker='o', markersize=3)
        plt.plot(self.timestamps, self.right_forces, 'r-', linewidth=2, label='Right Finger', marker='s', markersize=3)
        
        # Add threshold line
        plt.axhline(y=0.1, color='gray', linestyle='--', alpha=0.5, label='Contact Threshold (0.1N)')
        
        # Labels and formatting
        plt.xlabel('Timestep', fontsize=12)
        plt.ylabel('Force (N)', fontsize=12)
        phase_title = f" - {self.phase_name}" if self.phase_name else ""
        plt.title(f'Gripper Finger Forces{phase_title}', fontsize=14, fontweight='bold')
        plt.legend(loc='upper left', fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save to file
        output_path = os.path.join(output_dir, f"force_plot_{self.phase_name.replace(' ', '_')}.png")
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        print(f"[ForceViz] Saved plot to {output_path}")
        
        # Try to open with default image viewer
        if block:
            try:
                # Try to open with xdg-open (Linux), open (macOS), or start (Windows)
                if os.name == 'posix':
                    if os.path.exists('/usr/bin/xdg-open'):
                        subprocess.Popen(['xdg-open', output_path])
                    elif os.path.exists('/usr/bin/eog'):  # Eye of GNOME
                        subprocess.Popen(['eog', output_path])
                    elif os.path.exists('/usr/bin/feh'):  # feh image viewer
                        subprocess.Popen(['feh', output_path])
                print(f"[ForceViz] Opened plot: {output_path}")
                print("[ForceViz] Close the image viewer to resume execution...")
                # Wait for user to close the image (simple polling)
                import time
                time.sleep(2)
            except Exception as e:
                print(f"[ForceViz] Could not open image viewer: {e}")
                print(f"[ForceViz] Plot saved at: {output_path}")
    
    def summary(self) -> dict:
        """Return summary statistics.
        
        Returns:
            Dict with min/max/mean forces
        """
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
