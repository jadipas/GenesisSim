"""Quick sanity test of slipgen migration."""
from slipgen.task_pick_place import run_pick_place_sweep

if __name__ == "__main__":
    # Single config for quick test
    results = run_pick_place_sweep(
        display_video=False,
        mu_vals=(0.6,),
        fn_caps=(5.0,),
        disturb_levels=(0,)
    )
    print("Test results:", results)
