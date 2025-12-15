"""Slipgen-facing logger extending SensorDataLogger with metadata and slip metrics."""
from typing import Dict, Any, Optional
from collections import defaultdict
import numpy as np
from slipgen.force_viz import ForceVisualizer
from slipgen.contact_detection import DEFAULT_SLIP_THRESHOLD


class Logger:
    """Collects and stores sensor data for slip detection and dataset creation."""
    
    def __init__(self, slip_threshold: float = DEFAULT_SLIP_THRESHOLD, sim_dt: float = None):
        """Initialize the logger.
        
        Args:
            slip_threshold: Distance threshold for slip detection (meters). Default: 0.015m
            sim_dt: Simulation timestep duration (seconds). If None, must be set later via set_sim_dt()
        """
        self.visualizer = ForceVisualizer(title="Gripper Force Monitoring")
        self.slip_threshold = slip_threshold
        self.sim_dt = sim_dt
        self.reset()
    
    def reset(self):
        """Clear all logged data."""
        self.data = defaultdict(list)
        self.timestep = 0
        self.phase_markers = {}
        self.grasp_phase_contact = []
        self.current_phase: str | None = None
        self.cycle_count: int = 0
        # Distance-based slip tracking
        self.baseline_ee_cube_distance: Optional[float] = None
        self.slip_events: list[Dict[str, Any]] = []  # List of slip event records
        self.slip_active_phases: set[str] = set()  # Phases where slip detection is active
        # Keep visualizer alive across resets
        # Note: sim_dt is preserved across resets
    
    def set_sim_dt(self, dt: float):
        """Set the simulation timestep duration.
        
        Args:
            dt: Timestep duration in seconds (e.g., 0.01 for 100Hz)
        """
        self.sim_dt = dt

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
    
    def save(self, filepath: str, include_slip_mask: bool = True):
        """Save logged data to npz file.
        
        Args:
            filepath: Path to save the .npz file
            include_slip_mask: If True, includes 'slip_mask' array where 1 indicates
                              slip occurred at or before that timestep (sticky behavior)
        """
        save_dict = {}
        for key, values in self.data.items():
            if len(values) > 0 and isinstance(values[0], np.ndarray):
                save_dict[key] = np.array(values)
            else:
                save_dict[key] = values
        
        # Add metadata
        save_dict['sim_dt'] = self.sim_dt if self.sim_dt is not None else -1.0
        save_dict['slip_threshold'] = self.slip_threshold
        save_dict['total_timesteps'] = self.timestep
        
        # Add slip mask (sticky: once slip happens, all subsequent = 1)
        if include_slip_mask:
            save_dict['slip_mask'] = self.get_slip_mask(sticky=True)
            save_dict['slip_mask_instantaneous'] = self.get_slip_mask(sticky=False)
            first_slip = self.get_first_slip_timestep()
            save_dict['first_slip_timestep'] = first_slip if first_slip is not None else -1
        
        np.savez_compressed(filepath, **save_dict)
        print(f"Saved sensor data to {filepath}")
        print(f"  sim_dt={self.sim_dt}s, total_timesteps={self.timestep}, slip_threshold={self.slip_threshold}m")
        if include_slip_mask:
            slip_occurred = save_dict['first_slip_timestep'] != -1
            print(f"  Slip mask included: slip_occurred={slip_occurred}, first_slip_timestep={save_dict['first_slip_timestep']}")
    
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
    
    # -------------------------------------------------------------------------
    # Distance-based slip detection methods
    # -------------------------------------------------------------------------
    
    def capture_grasp_baseline(self, ee_cube_distance: float):
        """Capture baseline distance at secure grasp for distance-based slip detection.
        
        Call this at the end of the grasping phase when the object is securely held.
        
        Args:
            ee_cube_distance: Current distance between end-effector and cube (meters)
        """
        self.baseline_ee_cube_distance = ee_cube_distance
        print(f"[Slip Detection] Baseline EE-cube distance captured: {ee_cube_distance:.4f}m")
    
    def set_slip_active_phases(self, phases: list[str]):
        """Set which phases should have active slip detection.
        
        Args:
            phases: List of phase names, e.g., ["Lifting", "Transport"]
        """
        self.slip_active_phases = set(phases)
    
    def check_and_log_slip(self, slip_result: Dict[str, Any]) -> bool:
        """Check slip detection result and log if slip occurred.
        
        Call this during phases where slip detection is active.
        
        Args:
            slip_result: Result dict from detect_slip_by_distance()
        
        Returns:
            True if slip was detected, False otherwise
        """
        # Only log if slip actually detected
        if not slip_result.get('slip_detected', False):
            return False
        
        # Check if current phase should have slip detection
        if self.current_phase and self.current_phase not in self.slip_active_phases:
            return False
        
        slip_event = {
            'timestep': self.timestep,
            'phase': self.current_phase,
            'displacement': slip_result['displacement_from_baseline'],
            'current_distance': slip_result['current_ee_cube_distance'],
            'threshold': slip_result['slip_threshold'],
            'vertical_slip': slip_result.get('vertical_slip', 0.0),
            'horizontal_slip': slip_result.get('horizontal_slip', 0.0),
        }
        self.slip_events.append(slip_event)
        
        # Also store in main data log for per-timestep analysis
        self.data['slip_detected'].append(True)
        self.data['slip_displacement'].append(slip_result['displacement_from_baseline'])
        
        return True
    
    def log_slip_check(self, slip_result: Dict[str, Any]):
        """Log slip check result (even if no slip detected) for analysis.
        
        Args:
            slip_result: Result dict from detect_slip_by_distance()
        """
        # Always log displacement for analysis, regardless of threshold
        if 'slip_detected' not in self.data:
            self.data['slip_detected'] = []
            self.data['slip_displacement'] = []
        
        self.data['slip_detected'].append(slip_result.get('slip_detected', False))
        self.data['slip_displacement'].append(slip_result.get('displacement_from_baseline', 0.0))
    
    def get_slip_events(self) -> list[Dict[str, Any]]:
        """Get all logged slip events."""
        return self.slip_events.copy()
    
    def is_slip_detection_active(self) -> bool:
        """Check if slip detection should be active for current phase."""
        if self.baseline_ee_cube_distance is None:
            return False
        if not self.slip_active_phases:
            return False
        return self.current_phase in self.slip_active_phases
    
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
    
    def get_slippage_metrics(self) -> Dict[str, Any]:
        """Get detailed slippage metrics for the pick-and-place task.
        
        Includes both force-based (legacy) and distance-based slip detection results.
        """
        metrics = {
            # Force-based (legacy) metrics
            'slippage_occurred': False,
            'grasp_to_drop_contact_loss': 0,
            'transport_phase_contact_pct': 100.0,
            'grasp_phase_contact_pct': 100.0,
            # Distance-based slip metrics
            'distance_slip_occurred': False,
            'distance_slip_count': 0,
            'distance_slip_events': [],
            'max_displacement': 0.0,
            'baseline_distance': self.baseline_ee_cube_distance,
        }
        
        # Force-based slip detection (legacy)
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
        
        # Distance-based slip detection metrics
        if self.slip_events:
            metrics['distance_slip_occurred'] = True
            metrics['distance_slip_count'] = len(self.slip_events)
            metrics['distance_slip_events'] = self.slip_events.copy()
            metrics['max_displacement'] = max(e['displacement'] for e in self.slip_events)
        
        # Also check logged slip_displacement data for max displacement
        if 'slip_displacement' in self.data and self.data['slip_displacement']:
            all_displacements = [d for d in self.data['slip_displacement'] if d is not None]
            if all_displacements:
                metrics['max_displacement'] = max(metrics['max_displacement'], max(all_displacements))
        
        return metrics
    
    def get_slip_mask(self, sticky: bool = True) -> np.ndarray:
        """Generate binary slip mask for the experiment.
        
        Args:
            sticky: If True (default), once slip occurs all subsequent timesteps 
                    are marked as 1 (slip state persists). If False, only the 
                    exact timesteps where slip was detected are marked.
        
        Returns:
            np.ndarray of shape (timestep,) with dtype=np.int32
            Values: 0 = no slip, 1 = slip occurred (or persisted if sticky=True)
        """
        mask = np.zeros(self.timestep, dtype=np.int32)
        
        if not self.slip_events:
            return mask
        
        # Get all timesteps where slip was first detected
        slip_timesteps = sorted(e['timestep'] for e in self.slip_events)
        
        if sticky:
            # Once slip happens, all subsequent timesteps are marked as 1
            first_slip = slip_timesteps[0]
            mask[first_slip:] = 1
        else:
            # Only mark exact timesteps where slip was detected
            for ts in slip_timesteps:
                if ts < len(mask):
                    mask[ts] = 1
        
        return mask
    
    def get_first_slip_timestep(self) -> Optional[int]:
        """Get the timestep when slip first occurred.
        
        Returns:
            Timestep index (int) or None if no slip occurred
        """
        if not self.slip_events:
            return None
        return min(e['timestep'] for e in self.slip_events)
    
    def save_force_plot(self, output_dir: str = ".", filename: str = "force_plot.png"):
        """Generate and save force plot from collected data."""
        self.visualizer.generate_and_save_plot(output_dir, filename)
    
    def reset_visualizer(self):
        """Reset visualizer for a new experiment."""
        self.visualizer = ForceVisualizer(title="Gripper Force Monitoring")
