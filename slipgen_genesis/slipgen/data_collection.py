"""Unified sensor data collection."""
from typing import Dict, Any
from slipgen.proprioception import get_proprioception
from slipgen.contact_detection import detect_contact_with_object
from slipgen.object_state import get_object_state


def collect_sensor_data(franka, end_effector, cube, cam, 
                        include_vision: bool = False,
                        include_contact: bool = False,
                        include_object_state: bool = False) -> Dict[str, Any]:
    """Collect sensor data for current timestep.
    
    Args:
        franka: Robot instance
        end_effector: End-effector link
        cube: Object being manipulated
        cam: Camera instance
        include_vision: Include RGB and depth images
        include_contact: Include force-based contact detection data
        include_object_state: Include object pose and velocity
    
    Returns:
        Dict with sensor data. Always includes proprioception.
        Contact and object state are collected internally for slip detection
        but only logged if their respective flags are True.
    """
    data = {}
    
    # Always log proprioception
    proprio = get_proprioception(franka, end_effector)
    data.update(proprio)
    
    # Contact detection - always collected for internal use, optionally logged
    contact = detect_contact_with_object(franka, end_effector, cube)
    if include_contact:
        data.update(contact)
    else:
        # Only keep minimal fields needed for slip detection internals
        data['in_contact'] = contact['in_contact']  # Needed for legacy contact lost detection
        data['cube_ee_distance'] = contact['cube_ee_distance']  # Needed for slip detection
    
    # Object state - always collected for internal use, optionally logged  
    obj_state = get_object_state(cube)
    if include_object_state:
        data.update(obj_state)
    else:
        # Only keep obj_pos for slip detection (not logged but used internally)
        data['obj_pos'] = obj_state['obj_pos']
    
    if include_vision:
        rgb, depth, seg, normal = cam.render(depth=True)
        data['rgb'] = rgb
        data['depth'] = depth
    
    return data
