#!/usr/bin/env python3
"""Inspect a sensor_data.npz file and print statistics with example data points.

Supports the new multi-instance format where each pick-and-place cycle is a separate instance.
"""
import argparse
import numpy as np
from pathlib import Path


def print_section(title: str):
    """Print a section header."""
    print(f"\n{'='*60}")
    print(f" {title}")
    print('='*60)


def format_array(arr, max_items: int = 6) -> str:
    """Format array for display, truncating if needed."""
    if arr is None:
        return "None"
    arr = np.asarray(arr)
    if arr.ndim == 0:
        return str(arr.item())
    if arr.size <= max_items:
        return np.array2string(arr, precision=4, suppress_small=True)
    # Show first and last few items
    half = max_items // 2
    if arr.ndim == 1:
        return f"[{', '.join(f'{x:.4f}' for x in arr[:half])}, ..., {', '.join(f'{x:.4f}' for x in arr[-half:])}]"
    else:
        return f"shape {arr.shape}, dtype {arr.dtype}"


def compute_dataset_statistics(data: dict, num_instances: int, sim_dt: float) -> dict:
    """Compute comprehensive statistics across all instances."""
    stats = {
        'timesteps_per_instance': [],
        'first_slip_timesteps': [],
        'slip_ratios': [],  # fraction of timesteps after slip per instance
        'max_ee_heights': [],
        'ee_travel_distances': [],
        'max_gripper_widths': [],
        'min_gripper_widths': [],
        'max_cube_ee_distances': [],
        'obj_z_ranges': [],
        'slip_occurred': [],
    }
    
    for i in range(num_instances):
        prefix = f"instance_{i}_"
        
        # Timesteps
        ts = int(data.get(f'{prefix}timesteps', 0))
        stats['timesteps_per_instance'].append(ts)
        
        # Slip info
        first_slip = data.get(f'{prefix}first_slip_timestep', -1)
        if first_slip is not None:
            first_slip = int(first_slip)
        else:
            first_slip = -1
        stats['first_slip_timesteps'].append(first_slip)
        stats['slip_occurred'].append(first_slip >= 0)
        
        # Slip ratio (fraction of instance spent in slip state)
        if first_slip >= 0 and ts > 0:
            slip_ratio = (ts - first_slip) / ts
        else:
            slip_ratio = 0.0
        stats['slip_ratios'].append(slip_ratio)
        
        # EE position stats
        ee_pos = data.get(f'{prefix}ee_pos', None)
        if ee_pos is not None:
            ee_pos = np.asarray(ee_pos)
            if ee_pos.ndim == 2 and ee_pos.shape[0] > 0:
                stats['max_ee_heights'].append(np.max(ee_pos[:, 2]))
                # Travel distance (sum of consecutive displacements)
                if len(ee_pos) > 1:
                    diffs = np.diff(ee_pos, axis=0)
                    travel = np.sum(np.linalg.norm(diffs, axis=1))
                    stats['ee_travel_distances'].append(travel)
        
        # Gripper width stats
        gw = data.get(f'{prefix}gripper_width', None)
        if gw is not None:
            gw = np.asarray(gw)
            if gw.size > 0:
                stats['max_gripper_widths'].append(np.max(gw))
                stats['min_gripper_widths'].append(np.min(gw))
        
        # Cube-EE distance
        cube_ee = data.get(f'{prefix}cube_ee_distance', None)
        if cube_ee is not None:
            cube_ee = np.asarray(cube_ee)
            if cube_ee.size > 0:
                stats['max_cube_ee_distances'].append(np.max(cube_ee))
        
        # Object Z range (lift height)
        obj_pos = data.get(f'{prefix}obj_pos', None)
        if obj_pos is not None:
            obj_pos = np.asarray(obj_pos)
            if obj_pos.ndim == 2 and obj_pos.shape[0] > 0:
                z_range = np.max(obj_pos[:, 2]) - np.min(obj_pos[:, 2])
                stats['obj_z_ranges'].append(z_range)
    
    return stats


def print_statistics(stats: dict, sim_dt: float, slip_threshold: float):
    """Print comprehensive dataset statistics."""
    print_section("DATASET STATISTICS")
    
    n = len(stats['timesteps_per_instance'])
    
    # --- Instance Count & Duration ---
    print("\n  📊 Instance & Duration Statistics")
    print(f"     Total instances:        {n}")
    
    ts_arr = np.array(stats['timesteps_per_instance'])
    print(f"     Timesteps per instance:")
    print(f"       Mean:   {np.mean(ts_arr):.1f}")
    print(f"       Std:    {np.std(ts_arr):.1f}")
    print(f"       Min:    {np.min(ts_arr)}")
    print(f"       Max:    {np.max(ts_arr)}")
    print(f"       Median: {np.median(ts_arr):.0f}")
    
    total_ts = np.sum(ts_arr)
    print(f"     Total timesteps:        {total_ts}")
    if sim_dt > 0:
        durations = ts_arr * sim_dt
        print(f"     Duration per instance:")
        print(f"       Mean:   {np.mean(durations):.2f} s")
        print(f"       Min:    {np.min(durations):.2f} s")
        print(f"       Max:    {np.max(durations):.2f} s")
        print(f"     Total dataset duration: {total_ts * sim_dt:.2f} s ({total_ts * sim_dt / 60:.1f} min)")
    
    # --- Slip Statistics ---
    print("\n  🎯 Slip Detection Statistics")
    slip_arr = np.array(stats['slip_occurred'])
    slip_count = np.sum(slip_arr)
    no_slip_count = n - slip_count
    print(f"     Instances with slip:    {slip_count}/{n} ({100*slip_count/n:.1f}%)")
    print(f"     Instances without slip: {no_slip_count}/{n} ({100*no_slip_count/n:.1f}%)")
    print(f"     Class balance ratio:    {slip_count}:{no_slip_count}")
    
    if slip_count > 0:
        # Time-to-slip statistics (only for instances with slip)
        slip_times = [t for t in stats['first_slip_timesteps'] if t >= 0]
        slip_times_arr = np.array(slip_times)
        print(f"     Time to first slip (timesteps):")
        print(f"       Mean:   {np.mean(slip_times_arr):.1f}")
        print(f"       Std:    {np.std(slip_times_arr):.1f}")
        print(f"       Min:    {np.min(slip_times_arr)}")
        print(f"       Max:    {np.max(slip_times_arr)}")
        if sim_dt > 0:
            print(f"     Time to first slip (seconds):")
            print(f"       Mean:   {np.mean(slip_times_arr) * sim_dt:.3f} s")
            print(f"       Min:    {np.min(slip_times_arr) * sim_dt:.3f} s")
            print(f"       Max:    {np.max(slip_times_arr) * sim_dt:.3f} s")
        
        # Slip ratio (how much of each instance is in slip state)
        slip_ratios = [r for r, s in zip(stats['slip_ratios'], stats['slip_occurred']) if s]
        if slip_ratios:
            print(f"     Slip ratio (slip_timesteps/total, for slip instances):")
            print(f"       Mean:   {np.mean(slip_ratios)*100:.1f}%")
            print(f"       Min:    {np.min(slip_ratios)*100:.1f}%")
            print(f"       Max:    {np.max(slip_ratios)*100:.1f}%")
    
    print(f"     Slip threshold used:    {slip_threshold:.4f} m ({slip_threshold*100:.1f} cm)")
    
    # --- Motion Statistics ---
    print("\n  🤖 Robot Motion Statistics")
    
    if stats['max_ee_heights']:
        heights = np.array(stats['max_ee_heights'])
        print(f"     Max EE height reached:")
        print(f"       Mean:   {np.mean(heights):.4f} m")
        print(f"       Min:    {np.min(heights):.4f} m")
        print(f"       Max:    {np.max(heights):.4f} m")
    
    if stats['ee_travel_distances']:
        travel = np.array(stats['ee_travel_distances'])
        print(f"     EE travel distance per instance:")
        print(f"       Mean:   {np.mean(travel):.4f} m")
        print(f"       Min:    {np.min(travel):.4f} m")
        print(f"       Max:    {np.max(travel):.4f} m")
        print(f"       Total:  {np.sum(travel):.2f} m")
    
    if stats['min_gripper_widths'] and stats['max_gripper_widths']:
        min_gw = np.array(stats['min_gripper_widths'])
        max_gw = np.array(stats['max_gripper_widths'])
        print(f"     Gripper width:")
        print(f"       Min (closed): {np.mean(min_gw):.4f} m (mean)")
        print(f"       Max (open):   {np.mean(max_gw):.4f} m (mean)")
    
    # --- Object Statistics ---
    print("\n  📦 Object Statistics")
    
    if stats['obj_z_ranges']:
        z_ranges = np.array(stats['obj_z_ranges'])
        print(f"     Object lift height (Z range):")
        print(f"       Mean:   {np.mean(z_ranges):.4f} m ({np.mean(z_ranges)*100:.1f} cm)")
        print(f"       Min:    {np.min(z_ranges):.4f} m")
        print(f"       Max:    {np.max(z_ranges):.4f} m")
    
    if stats['max_cube_ee_distances']:
        max_dists = np.array(stats['max_cube_ee_distances'])
        print(f"     Max cube-EE distance:")
        print(f"       Mean:   {np.mean(max_dists):.4f} m")
        print(f"       Max:    {np.max(max_dists):.4f} m")
    
    # --- Data Quality ---
    print("\n  ✅ Data Quality")
    ts_variance = np.std(ts_arr) / np.mean(ts_arr) * 100 if np.mean(ts_arr) > 0 else 0
    print(f"     Timestep consistency:   {100-ts_variance:.1f}% (lower variance = more consistent)")
    print(f"     Instances with data:    {n}/{n} (100%)")


def inspect_instance(data: dict, instance_idx: int, prefix: str, sim_dt: float, example_timestep: int = None):
    """Inspect a single instance within the dataset."""
    print(f"\n--- Instance {instance_idx} ---")
    
    # Instance metadata
    timesteps = data.get(f'{prefix}timesteps', None)
    if timesteps is not None:
        timesteps = int(timesteps)
        print(f"  Timesteps: {timesteps}")
        if sim_dt > 0:
            print(f"  Duration:  {timesteps * sim_dt:.2f} s")
    
    first_slip = data.get(f'{prefix}first_slip_timestep', None)
    if first_slip is not None:
        first_slip = int(first_slip)
        if first_slip >= 0:
            print(f"  First slip: timestep {first_slip}", end="")
            if sim_dt > 0:
                print(f" ({first_slip * sim_dt:.3f} s)")
            else:
                print()
        else:
            print(f"  First slip: None (no slip)")
    
    # Slip mask
    slip_mask = data.get(f'{prefix}slip_mask', None)
    if slip_mask is not None:
        slip_mask = np.asarray(slip_mask)
        slip_count = np.sum(slip_mask)
        slip_pct = 100 * slip_count / len(slip_mask) if len(slip_mask) > 0 else 0
        print(f"  Slip mask: {slip_count}/{len(slip_mask)} ({slip_pct:.1f}% slipped)")
    
    # Sensor data shapes
    sensor_keys = ['q', 'dq', 'tau', 'ee_pos', 'ee_quat', 'ee_lin_vel', 'gripper_width', 
                   'obj_pos', 'cube_ee_distance', 'in_contact', 'slip_displacement']
    
    print(f"  Sensor data shapes:")
    for key in sensor_keys:
        arr = data.get(f'{prefix}{key}', None)
        if arr is not None:
            arr = np.asarray(arr)
            if arr.size > 0:
                print(f"    {key:20s}: {arr.shape}")
    
    # Example data point (optional)
    if example_timestep is not None and timesteps is not None:
        idx = min(example_timestep, timesteps - 1)
        print(f"\n  Example at timestep {idx}:")
        for key in ['ee_pos', 'obj_pos', 'gripper_width']:
            arr = data.get(f'{prefix}{key}', None)
            if arr is not None:
                arr = np.asarray(arr)
                if arr.ndim >= 1 and idx < len(arr):
                    print(f"    {key}: {format_array(arr[idx])}")


def inspect_npz(filepath: str, instance_idx: int = None, example_timestep: int = None, 
                verbose: bool = False, print_all: bool = False):
    """Load and inspect an npz file."""
    path = Path(filepath)
    if not path.exists():
        print(f"Error: File not found: {filepath}")
        return
    
    print(f"Loading: {filepath}")
    data = dict(np.load(filepath, allow_pickle=True))
    
    keys = list(data.keys())
    print(f"Found {len(keys)} arrays in file")
    
    # Check if this is the new multi-instance format
    num_instances = data.get('num_instances', None)
    
    if num_instances is not None:
        # NEW MULTI-INSTANCE FORMAT
        num_instances = int(num_instances)
        print_section(f"MULTI-INSTANCE DATASET ({num_instances} instances)")
        
        sim_dt = float(data.get('sim_dt', 0.01))
        slip_threshold = float(data.get('slip_threshold', 0.05))
        
        print(f"  sim_dt:         {sim_dt:.6f} s ({1/sim_dt:.1f} Hz)")
        print(f"  slip_threshold: {slip_threshold:.4f} m ({slip_threshold*100:.1f} cm)")
        print(f"  num_instances:  {num_instances}")
        
        # Compute and print comprehensive statistics
        stats = compute_dataset_statistics(data, num_instances, sim_dt)
        print_statistics(stats, sim_dt, slip_threshold)
        
        # Per-instance summary table (only if --print-all)
        if print_all:
            print_section("INSTANCE SUMMARY (all instances)")
            
            print(f"  {'Idx':>4s}  {'Timesteps':>10s}  {'Duration':>10s}  {'Slip':>8s}  {'First Slip':>12s}")
            print(f"  {'-'*4}  {'-'*10}  {'-'*10}  {'-'*8}  {'-'*12}")
            
            for i in range(num_instances):
                prefix = f"instance_{i}_"
                ts = int(data.get(f'{prefix}timesteps', 0))
                duration = f"{ts * sim_dt:.2f}s" if sim_dt > 0 else "N/A"
                first_slip = data.get(f'{prefix}first_slip_timestep', -1)
                if first_slip is not None:
                    first_slip = int(first_slip)
                else:
                    first_slip = -1
                
                slip_str = "Yes" if first_slip >= 0 else "No"
                first_slip_str = str(first_slip) if first_slip >= 0 else "-"
                
                print(f"  {i:>4d}  {ts:>10d}  {duration:>10s}  {slip_str:>8s}  {first_slip_str:>12s}")
        else:
            print(f"\n  (Use --print-all to see per-instance table)")
        
        # Detailed instance inspection
        if instance_idx is not None:
            if 0 <= instance_idx < num_instances:
                print_section(f"DETAILED INSTANCE {instance_idx}")
                prefix = f"instance_{instance_idx}_"
                inspect_instance(data, instance_idx, prefix, sim_dt, example_timestep)
            else:
                print(f"\nWarning: Instance {instance_idx} out of range [0, {num_instances-1}]")
        elif verbose:
            # Show all instances in detail
            for i in range(num_instances):
                prefix = f"instance_{i}_"
                inspect_instance(data, i, prefix, sim_dt, example_timestep)
        
        # List all keys
        if verbose:
            print_section("ALL ARRAYS IN FILE")
            for key in sorted(keys):
                arr = np.asarray(data[key])
                if arr.ndim == 0:
                    print(f"  {key:40s} scalar: {arr.item()}")
                else:
                    print(f"  {key:40s} shape={str(arr.shape):15s}")
    
    else:
        # LEGACY SINGLE-INSTANCE FORMAT (continuous timesteps)
        print_section("LEGACY FORMAT (continuous timesteps)")
        print("  Note: This is the old format with continuous timesteps.")
        print("  Consider regenerating with the new multi-instance format.")
        
        # Use the old inspection logic
        inspect_legacy_format(data, example_timestep)


def inspect_legacy_format(data: dict, example_timestep: int = None):
    """Inspect old format with continuous timesteps."""
    
    sim_dt = data.get('sim_dt', None)
    if sim_dt is not None:
        sim_dt = float(sim_dt)
        print(f"  sim_dt:            {sim_dt:.6f} s" + (" (not set)" if sim_dt < 0 else f" ({1/sim_dt:.1f} Hz)"))
    
    total_timesteps = data.get('total_timesteps', None)
    if total_timesteps is not None:
        total_timesteps = int(total_timesteps)
        print(f"  total_timesteps:   {total_timesteps}")
    
    slip_threshold = data.get('slip_threshold', None)
    if slip_threshold is not None:
        print(f"  slip_threshold:    {float(slip_threshold):.4f} m")
    
    first_slip = data.get('first_slip_timestep', None)
    if first_slip is not None:
        first_slip = int(first_slip)
        if first_slip >= 0:
            print(f"  first_slip_timestep: {first_slip}")
        else:
            print(f"  first_slip_timestep: None (no slip)")
    
    # Slip mask
    slip_mask = data.get('slip_mask', None)
    if slip_mask is not None:
        slip_mask = np.asarray(slip_mask)
        slip_count = np.sum(slip_mask)
        print(f"  Slip: {slip_count}/{len(slip_mask)} timesteps")
    
    # List all arrays
    print_section("ALL ARRAYS")
    for key in sorted(data.keys()):
        arr = np.asarray(data[key])
        if arr.ndim == 0:
            print(f"  {key:30s} scalar: {arr.item()}")
        else:
            print(f"  {key:30s} shape={str(arr.shape):15s}")


def main():
    parser = argparse.ArgumentParser(
        description="Inspect a sensor_data.npz dataset file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic inspection (summary statistics)
  python inspect_dataset.py sensor_data.npz
  
  # Show per-instance table (can be verbose for large datasets)
  python inspect_dataset.py sensor_data.npz --print-all
  
  # Inspect specific instance in detail
  python inspect_dataset.py sensor_data.npz --instance 0
  
  # Show all instances in detail (very verbose)
  python inspect_dataset.py sensor_data.npz --verbose
  
  # Show example data at specific timestep within an instance
  python inspect_dataset.py sensor_data.npz -i 0 -t 100
        """
    )
    parser.add_argument("file", type=str, help="Path to the .npz file")
    parser.add_argument("-i", "--instance", type=int, default=None,
                        help="Inspect a specific instance in detail (0-indexed)")
    parser.add_argument("-t", "--timestep", type=int, default=None,
                        help="Show example data at specific timestep within instance")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Show detailed information for all instances")
    parser.add_argument("--print-all", action="store_true",
                        help="Print per-instance summary table (can be long for big datasets)")
    
    args = parser.parse_args()
    inspect_npz(args.file, args.instance, args.timestep, args.verbose, args.print_all)


if __name__ == "__main__":
    main()
