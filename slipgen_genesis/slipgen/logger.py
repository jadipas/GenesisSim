"""Slipgen-facing logger extending SensorDataLogger with metadata and slip metrics."""
from typing import Dict, Any, Optional, List
from collections import defaultdict
import numpy as np
from slipgen.force_viz import ForceVisualizer
from slipgen.contact_detection import DEFAULT_SLIP_THRESHOLD

# Genesis default simulation timestep
GENESIS_DEFAULT_DT = 0.01


class Logger:
    """Collects and stores sensor data for slip detection and dataset creation.
    
    Each pick-and-place cycle (from Hover to Releasing) is treated as a separate
    data instance. Instances are stored separately and saved with instance indices.
    """
    
    def __init__(self, slip_threshold: float = DEFAULT_SLIP_THRESHOLD, sim_dt: float = GENESIS_DEFAULT_DT):
        """Initialize the logger.
        
        Args:
            slip_threshold: Distance threshold for slip detection (meters). Default: 0.05m
            sim_dt: Simulation timestep duration (seconds). Default: 0.01 (Genesis default)
        """
        self.visualizer = ForceVisualizer(title="Gripper Force Monitoring")
        self.slip_threshold = slip_threshold
        self.sim_dt = sim_dt
        self.cycle_count = 0
        
        # Storage for completed instances
        self.instances: List[Dict[str, Any]] = []
        
        # Current instance being recorded
        self._reset_current_instance()
    
    def _reset_current_instance(self):
        """Reset the current instance data (called when starting a new pick-and-place)."""
        self.data = defaultdict(list)
        self.timestep = 0
        self.phase_markers = {}
        self.grasp_phase_contact = []
        self.current_phase: str | None = None
        # Distance-based slip tracking
        self.baseline_ee_cube_distance: Optional[float] = None
        self.slip_events: list[Dict[str, Any]] = []  # List of slip event records
        self.slip_active_phases: set[str] = set()  # Phases where slip detection is active
        self._instance_started = False
    
    def reset(self):
        """Clear all logged data including all instances."""
        self.instances = []
        self.cycle_count = 0
        self._reset_current_instance()
        # Keep visualizer alive across resets
    
    def set_sim_dt(self, dt: float):
        """Set the simulation timestep duration.
        
        Args:
            dt: Timestep duration in seconds (e.g., 0.01 for 100Hz)
        """
        self.sim_dt = dt
    
    def start_instance(self):
        """Mark the start of a new pick-and-place instance.
        
        Call this at the beginning of each pick-and-place cycle (Hover phase start).
        """
        if self._instance_started and self.timestep > 0:
            # Auto-finalize previous instance if not done explicitly
            self.end_instance()
        self._reset_current_instance()
        self._instance_started = True
        print(f"[Logger] Started new instance (will be instance #{len(self.instances)})")
    
    def end_instance(self):
        """Finalize the current pick-and-place instance.
        
        Call this at the end of each pick-and-place cycle (after Releasing phase).
        Stores the instance and prepares for the next one.
        """
        if not self._instance_started or self.timestep == 0:
            print("[Logger] Warning: end_instance() called without data - skipping")
            return
        
        # Build instance data dict
        instance_data = {
            'timesteps': self.timestep,
            'sim_dt': self.sim_dt,
            'slip_threshold': self.slip_threshold,
            'phase_markers': dict(self.phase_markers),
            'slip_events': list(self.slip_events),
            'baseline_ee_cube_distance': self.baseline_ee_cube_distance,
            'slip_mask': self.get_slip_mask(sticky=True),
            'slip_mask_instantaneous': self.get_slip_mask(sticky=False),
            'first_slip_timestep': self.get_first_slip_timestep(),
        }
        
        # Copy sensor data arrays
        for key, values in self.data.items():
            if len(values) > 0 and isinstance(values[0], np.ndarray):
                instance_data[key] = np.array(values)
            else:
                instance_data[key] = np.array(values) if values else np.array([])
        
        self.instances.append(instance_data)
        self.cycle_count += 1
        
        first_slip = instance_data['first_slip_timestep']
        slip_occurred = first_slip is not None
        print(f"[Logger] Finalized instance #{len(self.instances)-1}: "
              f"{self.timestep} timesteps, slip_occurred={slip_occurred}, first_slip={first_slip}")
        
        self._instance_started = False

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
        """Save all logged instances to npz file.
        
        Each instance is a separate pick-and-place cycle, saved with prefix 'instance_N_'.
        Also saves global metadata.
        
        Args:
            filepath: Path to save the .npz file
            include_slip_mask: If True, includes slip_mask arrays per instance
        """
        # Finalize current instance if still recording
        if self._instance_started and self.timestep > 0:
            self.end_instance()
        
        if len(self.instances) == 0:
            print(f"[Logger] Warning: No instances to save!")
            return
        
        save_dict = {}
        
        # Global metadata
        save_dict['num_instances'] = len(self.instances)
        save_dict['sim_dt'] = self.sim_dt
        save_dict['slip_threshold'] = self.slip_threshold
        
        # Per-instance data with prefix
        for idx, instance in enumerate(self.instances):
            prefix = f"instance_{idx}_"
            
            # Metadata for this instance
            save_dict[f'{prefix}timesteps'] = instance['timesteps']
            save_dict[f'{prefix}first_slip_timestep'] = instance['first_slip_timestep'] if instance['first_slip_timestep'] is not None else -1
            
            # Slip masks
            if include_slip_mask:
                save_dict[f'{prefix}slip_mask'] = instance['slip_mask']
                save_dict[f'{prefix}slip_mask_instantaneous'] = instance['slip_mask_instantaneous']
            
            # Sensor data arrays
            for key, values in instance.items():
                if key in ('timesteps', 'sim_dt', 'slip_threshold', 'phase_markers', 
                          'slip_events', 'baseline_ee_cube_distance', 'slip_mask', 
                          'slip_mask_instantaneous', 'first_slip_timestep'):
                    continue  # Skip metadata, already handled
                
                if isinstance(values, np.ndarray) and values.size > 0:
                    save_dict[f'{prefix}{key}'] = values
        
        np.savez_compressed(filepath, **save_dict)
        
        print(f"\nSaved {len(self.instances)} instances to {filepath}")
        print(f"  Global: sim_dt={self.sim_dt}s, slip_threshold={self.slip_threshold}m")
        
        # Summary per instance
        for idx, instance in enumerate(self.instances):
            slip_occurred = instance['first_slip_timestep'] is not None
            # print(f"  Instance {idx}: {instance['timesteps']} timesteps, slip={slip_occurred}")
    
    def get_instance_count(self) -> int:
        """Get number of completed instances."""
        return len(self.instances)
    
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
