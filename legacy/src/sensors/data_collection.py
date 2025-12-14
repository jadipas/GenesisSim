"""Unified sensor data collection."""
from typing import Dict, Any
from .proprioception import get_proprioception
from .contact_detection import detect_contact_with_object
from .object_state import get_object_state


def collect_sensor_data(franka, end_effector, cube, cam, include_vision: bool = False) -> Dict[str, Any]:
    """
    Collect all sensor data for current timestep.
    
    Args:
        franka: Robot entity
        end_effector: End-effector link
        cube: Cube entity
        cam: Camera object
        include_vision: Whether to capture and include RGBD images
    
    Returns:
        Dict with all sensor readings
    """
    data = {}
    
    proprio = get_proprioception(franka, end_effector)
    data.update(proprio)
    
    contact = detect_contact_with_object(franka, end_effector, cube)
    data.update(contact)
    
    obj_state = get_object_state(cube)
    data.update(obj_state)
    
    if include_vision:
        rgb, depth, seg, normal = cam.render(depth=True)
        data['rgb'] = rgb
        data['depth'] = depth
    
    return data
