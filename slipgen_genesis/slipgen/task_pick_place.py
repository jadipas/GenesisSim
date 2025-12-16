"""Pick-and-place task API built on migrated demo/steps with slip knobs."""
from typing import List, Optional
import numpy as np
import tempfile
import subprocess
import os
import atexit

from slipgen.demo import run_iterative_pick_and_place, run_pick_and_place_demo
from slipgen.logger import Logger
from slipgen.knobs import SlipKnobs
from slipgen.scene import setup_with_knobs, reset_cube_positions, scatter_cubes


class ProgressBarWindow:
    """A progress bar that runs in a separate terminal window."""
    
    def __init__(self, total: int, desc: str = "Progress"):
        self.total = total
        self.current = 0
        self.desc = desc
        self._progress_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.progress')
        self._progress_file.write("0\n")
        self._progress_file.flush()
        self._progress_path = self._progress_file.name
        self._process = None
        self._start_terminal()
        atexit.register(self.close)
    
    def _start_terminal(self):
        """Launch a separate terminal window with a progress bar script."""
        # Python script that reads progress from file and displays a bar
        script = f'''
import sys
import time

total = {self.total}
desc = "{self.desc}"
progress_file = "{self._progress_path}"

def draw_bar(current, total, width=40):
    pct = current / total if total > 0 else 0
    filled = int(width * pct)
    bar = "█" * filled + "░" * (width - filled)
    return f"\\r{{desc}}: |{{bar}}| {{current}}/{{total}} ({{pct*100:.1f}}%)"

print("\\033[92m" + "=" * 60)
print(f"  {{desc}} - Progress Monitor")
print("=" * 60 + "\\033[0m")
print()

try:
    while True:
        try:
            with open(progress_file, 'r') as f:
                content = f.read().strip()
                if content == "DONE":
                    print(draw_bar(total, total), end="")
                    print("\\n\\n\\033[92m✓ Complete!\\033[0m")
                    break
                current = int(content)
                print(draw_bar(current, total), end="", flush=True)
        except (ValueError, FileNotFoundError):
            pass
        time.sleep(0.1)
except KeyboardInterrupt:
    pass

print("\\n\\nPress Enter to close...")
input()
'''
        # Write the script to a temp file
        self._script_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.py')
        self._script_file.write(script)
        self._script_file.close()
        
        # Try different terminal emulators
        terminals = [
            ['gnome-terminal', '--', 'python3', self._script_file.name],
            ['xterm', '-hold', '-e', 'python3', self._script_file.name],
            ['konsole', '-e', 'python3', self._script_file.name],
            ['xfce4-terminal', '-e', f'python3 {self._script_file.name}'],
        ]
        
        for cmd in terminals:
            try:
                self._process = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                break
            except FileNotFoundError:
                continue
        
        if self._process is None:
            print("[ProgressBar] Warning: Could not open a separate terminal window. Progress will be printed inline.")
    
    def update(self, n: int = 1):
        """Update progress by n samples."""
        self.current += n
        try:
            with open(self._progress_path, 'w') as f:
                f.write(f"{self.current}\n")
        except Exception:
            pass
    
    def close(self):
        """Signal completion and cleanup."""
        try:
            with open(self._progress_path, 'w') as f:
                f.write("DONE\n")
        except Exception:
            pass
        # Give the terminal a moment to read the DONE signal
        import time
        time.sleep(0.2)
        # Cleanup temp files
        try:
            os.unlink(self._progress_path)
        except Exception:
            pass
        try:
            os.unlink(self._script_file.name)
        except Exception:
            pass


def run_pick_place_sweep(num_cubes: int = 3, show_viewer: bool = True, render_cameras: bool = True,
                         mu_vals=(0.2, 0.6, 1.0), q_targets=(-0.02, -0.03, -0.04), disturb_levels=(0, 1, 2)):
    """Run parameter sweep with scene reuse for efficiency.
    
    Args:
        num_cubes: Number of cubes in the scene
        show_viewer: Show Genesis 3D viewer window
        render_cameras: Render OpenCV RGB/depth camera windows
        mu_vals: Friction coefficient values to sweep
        fn_caps: Force cap values to sweep
        disturb_levels: Disturbance levels to sweep
    """
    from slipgen.scene import init_scene, apply_knobs
    
    # One-time expensive scene initialization
    print("Initializing scene (one-time build)...")
    print(f"  show_viewer={show_viewer}, render_cameras={render_cameras}")
    cube_area = {
        "x_range": (0.35, 0.55),
        "y_range": (-0.55, -0.05),
        "z": 0.035,
        "min_separation": 0.08,
    }
    scene, franka, cam, end_effector, cubes, motors_dof, fingers_dof = init_scene(show_viewer=show_viewer, cube_area=cube_area)
    results = []
    total_configs = len(mu_vals) * len(q_targets) * len(disturb_levels)
    config_idx = 0
    
    for mu in mu_vals:
        for q in q_targets:
            for d in disturb_levels:
                config_idx += 1
                print(f"\n[{config_idx}/{total_configs}] Testing config: mu={mu}, q_finger_target={q}, disturb={d}")
                
                knobs = SlipKnobs(mu=mu, disturb_level=d, q_finger_target=q)
                apply_knobs(knobs, scene, franka, cubes, fingers_dof)
                
                logger = Logger()
                run_iterative_pick_and_place(franka, scene, cam, end_effector, cubes, logger, motors_dof, fingers_dof, 
                                            show_viewer=show_viewer, render_cameras=render_cameras, knobs=knobs)
                stats = logger.get_slippage_metrics()
                results.append({'mu': mu, 'q_finger_target': q, 'disturb': d, **stats})
                print(f"  Result: {stats}")
                
                # Save force plot for this configuration
                filename = f"force_plot_sweep_mu{mu}_q{q}_d{d}.png"
                # logger.save_force_plot(output_dir=".", filename=filename)
                reset_cube_positions(cubes, cube_area)
    
    print(f"\nSweep finished: tested {total_configs} configurations.")
    return results


def generate_dataset(num_samples: int = 10, show_viewer: bool = False, render_cameras: bool = False,
                     mu: float = 0.6, fn_cap: float = 5.0, q_finger_target: float = -0.03, disturb_level: int = 1, min_cubes=3, max_cubes=5,
                     save_path: str = "sensor_data.npz"):
    """Generate dataset from pick-and-place runs.
    
    Args:
        num_samples: Number of pick-and-place cycles
        show_viewer: Show Genesis 3D viewer window
        render_cameras: Render OpenCV RGB/depth camera windows
        mu: Friction coefficient
        fn_cap: Force cap
        disturb_level: Disturbance level
        save_path: Path to save the sensor data
    """
    knobs = SlipKnobs(mu=mu, fn_cap=fn_cap, disturb_level=disturb_level, q_finger_target=q_finger_target)
    cube_area = {
        "x_range": (0.35, 0.55),
        "y_range": (-0.55, -0.05),
        "z": 0.035,
        "min_separation": 0.08,
    }
    scene, franka, cam, end_effector, cubes, motors_dof, fingers_dof = setup_with_knobs(knobs, show_viewer=show_viewer, cube_area=cube_area, num_cubes=max_cubes)
    scatter_cubes(cubes)
    logger = Logger()
    
    # Launch progress bar in a separate terminal window
    pbar = ProgressBarWindow(total=num_samples, desc="Generating samples")
    
    samples_generated = 0
    remaining = num_samples
    while remaining > 0:
        sample = np.random.randint(min_cubes, max_cubes + 1)
        sample = min(sample, remaining)  # Don't exceed requested samples
        remaining -= sample
        scubes = cubes[:sample]
        reset_cube_positions(scubes, cube_area)
        drop_pos = np.array([0.55, 0.38, 0.14])
        run_iterative_pick_and_place(franka, scene, cam, end_effector, scubes, logger, motors_dof, fingers_dof,
                                show_viewer=show_viewer, render_cameras=render_cameras, knobs=knobs)
        samples_generated += sample
        pbar.update(sample)
    
    pbar.close()
    logger.save(save_path)
    
    # Save final force plot for entire dataset
    # logger.save_force_plot(output_dir=".", filename="dataset_force_plot.png")
