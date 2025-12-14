"""Centralized knobs/config parameters for tasks and sweeps."""
from dataclasses import dataclass
from typing import Tuple

@dataclass
class PickPlaceKnobs:
    # Task geometry
    hover_offset: float = 0.20
    lift_offset: float = 0.15
    arc_height_min: float = 0.55
    arc_height_max: float = 0.65
    default_drop: Tuple[float, float, float] = (0.55, 0.38, 0.14)

@dataclass
class SlipKnobs:
    # Core slip generator knobs
    mu: float = 0.6            # contact friction
    fn_cap: float = 5.0        # max normal force per finger [N]
    disturb_level: int = 0     # 0=no bump, 1..3 increasing severity
