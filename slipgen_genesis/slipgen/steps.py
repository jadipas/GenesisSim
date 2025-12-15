"""Step-by-step execution functionality."""
import numpy as np
import cv2
from slipgen.camera import update_wrist_camera
from slipgen.data_collection import collect_sensor_data
from slipgen.contact_detection import detect_slip_by_distance


def execute_steps(franka, scene, cam, end_effector, cube, logger, 
                 num_steps, motors_dof=None, qpos=None, finger_force=None, fingers_dof=None, finger_qpos=None,
                 render_cameras=True, print_status=False, print_interval=20, phase_name="", debug=False, knobs=None,
                 check_slip: bool = True, slip_threshold: float = None):
    """Execute simulation steps with sensor data collection.
    
    Args:
        render_cameras: If True, render OpenCV RGB/depth windows. Set False for headless.
    
    If finger_qpos is provided, fingers are controlled via position targets with force cap.
    If finger_force is provided (legacy), a clamped holding force is applied.
    The force cap from knobs.fn_cap is always enforced.
    
    Distance-based slip detection:
        If check_slip=True and the logger has a baseline distance set (via logger.capture_grasp_baseline()),
        slip detection will run during phases marked as active (via logger.set_slip_active_phases()).
        Use slip_threshold to override the default threshold (0.015m).
    """
    if phase_name:
        print(f"{phase_name}...")
    
    if qpos is not None and motors_dof is not None:
        franka.control_dofs_position(qpos, motors_dof)
    
    # Extract force cap from knobs
    fn_cap = None
    if knobs is not None and hasattr(knobs, 'fn_cap'):
        fn_cap = float(knobs.fn_cap)
    
    # Clamp finger force to knob's fn_cap if knobs provided (only log once at start)
    clamped_force = finger_force
    clamp_logged = False
    if finger_force is not None and fn_cap is not None:
        clamped_force = np.clip(finger_force, -fn_cap, fn_cap)
        if not np.allclose(finger_force, clamped_force):
            print(f"  [Force Clamp] Requested {finger_force}, clamped to [{-fn_cap:.1f}, {fn_cap:.1f}] -> {clamped_force}")
            clamp_logged = True
    
    for i in range(num_steps):
        # Apply finger control
        if finger_qpos is not None and fingers_dof is not None:
            # Position control: move fingers to target position
            finger_target = np.asarray(finger_qpos, dtype=float)
            if finger_target.size == 1:
                finger_target = np.array([finger_target.item(), finger_target.item()])
            franka.control_dofs_position(finger_target, fingers_dof)
        elif clamped_force is not None and fingers_dof is not None:
            # Legacy force control path
            franka.control_dofs_force(clamped_force, fingers_dof)
        
        scene.step()
        update_wrist_camera(cam, end_effector)
        
        sensor_data = collect_sensor_data(franka, end_effector, cube, cam, include_vision=False)
        logger.log_step(sensor_data)
        
        # Distance-based slip detection
        if check_slip and logger.is_slip_detection_active():
            threshold = slip_threshold if slip_threshold is not None else logger.slip_threshold
            slip_result = detect_slip_by_distance(
                cube_pos=sensor_data['obj_pos'],
                ee_pos=sensor_data['ee_pos'],
                baseline_distance=logger.baseline_ee_cube_distance,
                threshold=threshold,
            )
            logger.log_slip_check(slip_result)
            if logger.check_and_log_slip(slip_result):
                print(f"SLIP DETECTED at timestep {logger.timestep}, displacement: {slip_result['displacement_from_baseline']:.4f}m (threshold: {threshold:.4f}m)")
        
        if debug and i % print_interval == 0:
            q_current = franka.get_qpos()
            dq_current = franka.get_dofs_velocity()
            tau_current = franka.get_dofs_force()
            if hasattr(q_current, 'cpu'):
                q_current = q_current.cpu().numpy()
            if hasattr(dq_current, 'cpu'):
                dq_current = dq_current.cpu().numpy()
            if hasattr(tau_current, 'cpu'):
                tau_current = tau_current.cpu().numpy()
            print(f"[{phase_name or 'Phase'} Step {i}]")
            print(f"  Target: {qpos if qpos is not None else 'N/A'}")
            print(f"  Joint pos: {q_current}")
            print(f"  Joint vel: {dq_current}")
            print(f"  Joint tau: {tau_current}")
            print(f"  EE pos: {sensor_data['ee_pos']}, EE vel: {sensor_data['ee_lin_vel']}")
        
        if print_status and i % print_interval == 0:
            contact_str = "IN CONTACT" if sensor_data.get('in_contact', False) else "NO CONTACT"
            cube_height = sensor_data['obj_pos'][2] if 'obj_pos' in sensor_data else 0.0
            # Handle case where contact forces aren't logged
            left_f = sensor_data.get('left_finger_force', None)
            right_f = sensor_data.get('right_finger_force', None)
            if left_f is not None and right_f is not None:
                force_str = f"L:{left_f:.3f}N R:{right_f:.3f}N"
            else:
                force_str = f"dist:{sensor_data.get('cube_ee_distance', 0):.4f}m"
            print(f"  Step {i}: {contact_str}, {force_str}, "
                  f"Gripper: {sensor_data['gripper_width']:.4f}m, Cube Z: {cube_height:.3f}m")
        
        if logger.detect_contact_lost():
            print(f"CONTACT LOST at timestep {logger.timestep}, Cube height: {sensor_data['obj_pos'][2]:.3f}m")
        
        if render_cameras and cam is not None:
            rgb, depth, seg, normal = cam.render(depth=True)
            depth_vis = (depth / depth.max() * 255).astype('uint8')
            cv2.imshow("RGB", rgb[:, :, ::-1])
            cv2.imshow("Depth", depth_vis)
            cv2.waitKey(1)
