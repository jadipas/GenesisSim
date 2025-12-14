"""Command line argument parsing."""
import argparse


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Genesis Franka Pick-and-Place Simulation')
    parser.add_argument('--headless', action='store_true', 
                       help='Run without viewer and without camera playback')
    parser.add_argument('--sim-only', action='store_true',
                       help='Run with viewer but without camera playback')
    return parser.parse_args()
