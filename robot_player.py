import json
import time
import sys
import signal
import math
import threading
import numpy as np

import pylibfranka as franka
from pylibfranka_examples.example_common import MotionGenerator, setDefaultBehaviour

from realsense_module import RealSenseTracker, get_object_in_world_frame


ROBOT_IP = "172.16.0.2"

GRIPPER_THRESHOLD = 0.002                       # 2mm buffer to ignore minor floating-point noise
home_q = [0, -0.5, 0, -2.5, 0, 2.0, 0.8]        # franka arm neutral pose

kDefaultMaximumVelocities = [0.655, 0.655, 0.655, 0.655, 1.315, 1.315, 1.315]
kDefaultGoalTolerance = 10.0
kStartJointTolerance = 0.05
kGripperMoveSpeed = 0.1                         # m/s
kGripperForce = 60.0                            # N

motion_finished = False


def _camera_stream_worker(camera, stop_event):
    try:
        camera.stream_loop(stop_event=stop_event)
    except Exception as exc:
        print(f"Camera stream stopped due to error: {exc}")
        stop_event.set()


def signal_handler(sig, frame):
    global motion_finished
    if sig == signal.SIGINT:
        motion_finished = True


def validate_trajectory(trajectory):
    if not isinstance(trajectory, list) or not trajectory:
        raise ValueError("Trajectory must be a non-empty list of waypoints.")

    for index, step_data in enumerate(trajectory):
        joints = step_data.get("joints")
        if joints is None:
            continue  # gripper-only waypoint, allowed
        if not isinstance(joints, list) or len(joints) != 7:
            raise ValueError(f"Waypoint {index} must contain 7 joint values.")


def assert_robot_is_at_start(robot, trajectory, tolerance=kStartJointTolerance):
    robot_state = robot.read_once()
    current_joints = list(robot_state.q)
    start_joints = trajectory[0]["joints"]
    joint_errors = [abs(current - target) for current, target in zip(current_joints, start_joints)]
    max_joint_error = max(joint_errors)

    if max_joint_error > tolerance:
        current_deg = [round(math.degrees(value), 2) for value in current_joints]
        start_deg = [round(math.degrees(value), 2) for value in start_joints]
        error_deg = [round(math.degrees(value), 2) for value in joint_errors]
        raise RuntimeError(
            "Robot is not at the trajectory start pose. "
            f"Max joint error is {max_joint_error:.4f} rad ({math.degrees(max_joint_error):.2f} deg), "
            f"which exceeds the tolerance of {tolerance:.4f} rad ({math.degrees(tolerance):.2f} deg).\n"
            f"Current joints (deg): {current_deg}\n"
            f"Start joints (deg):   {start_deg}\n"
            f"Absolute error (deg): {error_deg}\n"
            "Move the arm to the start pose before replaying this file."
        )



def move_robot_to_start_pose(robot, trajectory, controller, tolerance=kStartJointTolerance):

    start_joints = trajectory[0]["joints"]
    robot_state = robot.read_once()
    current_joints = list(robot_state.q)
    joint_errors = [abs(current - target) for current, target in zip(current_joints, start_joints)]
    max_joint_error = max(joint_errors)

    if max_joint_error <= tolerance:
        return

    print(
        "Robot is not at the trajectory start pose. "
        f"Moving to start pose first. Max joint error: {max_joint_error:.4f} rad "
        f"({math.degrees(max_joint_error):.2f} deg)."
    )


    # Interpolate slowly from current to start joints
    steps = 1000  # 20 seconds at 50Hz — slow and safe
    try:
        for i in range(steps):
            if motion_finished:
                break

            # Read feedback to check for errors
            target_feedback = controller.get_target_feedback()
            if target_feedback.error_message is not None:
                raise RuntimeError(f"Error in feedback during start pose move: {target_feedback.error_message}")
            
            # Interpolate each joint linearly towards the start pose
            loop_start = time.monotonic()
            alpha = i / max(steps - 1, 1)
            target = [c + alpha * (s - c) for c, s in zip(current_joints, start_joints)]
            next_target = franka.AsyncPositionControlHandler.JointPositionTarget(
                joint_positions=target
            )   # safety filters for the position control handler
            command_result = controller.set_joint_position_target(next_target)

            if command_result.error_message is not None:
                raise RuntimeError(f"Hardware rejected target: {command_result.error_message}")
            sleep_time = 0.020 - (time.monotonic() - loop_start)
            if sleep_time > 0:
                time.sleep(sleep_time)
    finally:
        controller.stop_control()

    assert_robot_is_at_start(robot, trajectory, tolerance=tolerance)


def run_hardware_execution(filename="path_data/example_path.json"):

    #=========== LOAD THE FILE =================
    with open(filename, 'r') as f:
        trajectory = json.load(f)

    validate_trajectory(trajectory)



    # ===== SETUP ROBOT CONFIGURATION AND SAFETY =======
    signal.signal(signal.SIGINT, signal_handler)

    try:
        robot = franka.Robot(ROBOT_IP, franka.RealtimeConfig.kIgnore)
    except Exception as e:
        print(f"Could not connect to robot: {e}")
        sys.exit(-1)

    # Connect gripper if the trajectory includes gripper state
    gripper = None
    try:
        gripper = franka.Gripper(ROBOT_IP)
        gripper.homing()
    except Exception as e:
        print(f"Could not connect to gripper: {e}")
        sys.exit(-1)

    setDefaultBehaviour(robot)

    # =========== CAMERA SETUP ================
    print("\nSetting up RealSense camera for tracking...")
    try:
        camera = RealSenseTracker()
        print("Camera setup complete. Starting execution in 3s...")
        time.sleep(3)
    except Exception as e:
        print(f"Failed to initialize RealSense camera: {e}")
        sys.exit(-1)

    T_flange_to_camera = camera.T_flange_to_camera
    robot_state = robot.read_once()
    T_base_to_flange = np.array(robot_state.O_T_EE).reshape((4, 4), order='F')

    # --- Usage inside your vision loop ---
    # camera_xyz = [x_c, y_c, z_c] # Pulled from your rs2_deproject_pixel_to_point function
    camera_xyz = camera.get_object_camera_xyz()
    print(f"Camera Frame Position: {camera_xyz}")
    if camera_xyz is not None:
        object_world_xyz = camera.get_object_in_world_frame(T_base_to_flange)
        print(f"Real-World Workspace Position: {object_world_xyz}")
    else:
        print("No object detected in the first camera frame.")

    camera_stop_event = threading.Event()
    camera_thread = threading.Thread(
        target=_camera_stream_worker,
        args=(camera, camera_stop_event),
        daemon=True,
    )
    camera_thread.start()

    position_control_handler = None
    gripper_thread = None  # track gripper thread to join before next action
    try:
        # =========== CONFIGURE THE SAFE CONTROLLER ================
        joint_position_control_configuration = \
            franka.AsyncPositionControlHandler.Configuration(
                maximum_joint_velocities=kDefaultMaximumVelocities,
                goal_tolerance=kDefaultGoalTolerance
        )
        result = franka.AsyncPositionControlHandler.configure(
            robot,
            joint_position_control_configuration
            )
        if result.error_message is not None:
            print(result.error_message)
            sys.exit(-1)

        position_control_handler = result.handler

        # =========== MOVE ROBOT TO START POSE ================
        move_robot_to_start_pose(
            robot,
            trajectory,
            position_control_handler
        )
        time.sleep(0.5)

        # =========== MAIN EXECUTION LOOP ================
        time_step = 0.020  # 50 Hz matching trajectory file

        print("Pre-flight check passed. Starting execution in 3s... Hold the E-Stop!")
        time.sleep(3)

        last_gripper_width = trajectory[0].get("gripper") if gripper is not None else None
        for step_data in trajectory:
            if motion_finished:
                print("Stop requested. Halting hardware execution.")
                break

            loop_start = time.monotonic()

            target_feedback = position_control_handler.get_target_feedback()
            if target_feedback.error_message is not None:
                print(f"Error in feedback: {target_feedback.error_message}")
                sys.exit(-1)


            #====== ARM CONTROL =======
            joints = step_data.get("joints")
            if joints is not None:
                next_target = franka.AsyncPositionControlHandler.JointPositionTarget(
                    joint_positions=joints
                )
                command_result = position_control_handler.set_joint_position_target(next_target)
                if command_result.error_message is not None:
                    print(f"Hardware rejected target: {command_result.error_message}")
                    sys.exit(-1)


            #====== GRIPPER CONTROL =======
            if gripper is not None:
                new_width = step_data.get("gripper")
                
                if new_width is not None:
                    # Check if the file is commanding the gripper to close/grasp
                    if abs(new_width - last_gripper_width) > GRIPPER_THRESHOLD:
                        print(f"Gripper action detected in file ({last_gripper_width}m -> {new_width}m). Moving gripper...")
                        gripper.move(new_width, kGripperMoveSpeed)
                        last_gripper_width = new_width
                        loop_start = time.monotonic()   # CRITICAL: Reset the loop timer here as well


            # ====== TIMING REGULATION ======
            sleep_time = time_step - (time.monotonic() - loop_start)
            if sleep_time > 0:
                time.sleep(sleep_time)

        if not motion_finished:
            print("Trajectory playback finished. Vision windows will stay open.")
            print("Press 'q' in a camera window or Ctrl+C in terminal to exit.")
            while not motion_finished and not camera_stop_event.is_set():
                if not camera_thread.is_alive():
                    break
                time.sleep(0.1)

    finally:
        camera_stop_event.set()
        camera.stop()
        camera_thread.join(timeout=2.0)

        # Wait for any in-progress gripper action to complete
        if gripper_thread is not None:
            gripper_thread.join()
        if position_control_handler is not None:
            position_control_handler.stop_control()

    print("Execution complete.")


if __name__ == "__main__":
    run_hardware_execution(filename="path_data/trajectory_from_planned_path.json")