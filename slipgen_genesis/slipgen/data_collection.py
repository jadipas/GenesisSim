"""Unified sensor data collection."""
from typing import Dict, Any
from slipgen.proprioception import get_proprioception
from slipgen.contact_detection import detect_contact_with_object
from slipgen.object_state import get_object_state


def collect_sensor_data(franka, end_effector, cube, cam, include_vision: bool = False) -> Dict[str, Any]:
    """Collect all sensor data for current timestep."""
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
