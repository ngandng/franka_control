
import json
import time
import sys
import signal
import math
import threading

import pylibfranka as franka
from pylibfranka_examples.example_common import MotionGenerator, setDefaultBehaviour


ROBOT_IP = "172.16.0.2"

kDefaultMaximumVelocities = [0.655, 0.655, 0.655, 0.655, 1.315, 1.315, 1.315]
kDefaultGoalTolerance = 10.0
kStartJointTolerance = 0.05
kGripperMoveSpeed = 0.1  # m/s

motion_finished = False


def signal_handler(sig, frame):
    global motion_finished
    if sig == signal.SIGINT:
        motion_finished = True


def validate_trajectory(trajectory):
    if not isinstance(trajectory, list) or not trajectory:
        raise ValueError("Trajectory must be a non-empty list of waypoints.")

    for index, step_data in enumerate(trajectory):
        joints = step_data.get("joints")
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


def move_robot_to_start_pose(robot, trajectory, tolerance=kStartJointTolerance):
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

    motion_generator = MotionGenerator(speed_factor=0.1, q_goal=start_joints)

    def move_to_start_callback(robot_state, duration):
        if motion_finished:
            output = franka.JointPositions(list(robot_state.q))
            output.motion_finished = True
            return output

        duration_sec = duration.to_sec() if hasattr(duration, "to_sec") else float(duration)
        return motion_generator(robot_state, duration_sec)

    try:
        robot.control(move_to_start_callback)
    except Exception as exc:
        raise RuntimeError(f"Failed to move robot to the trajectory start pose: {exc}") from exc

    assert_robot_is_at_start(robot, trajectory, tolerance=tolerance)


def run_hardware_execution(filename="path_data/example_path.json"):

    # load the path file
    with open(filename, 'r') as f:
        trajectory = json.load(f)

    validate_trajectory(trajectory)


    signal.signal(signal.SIGINT, signal_handler)

    try:
        robot = franka.Robot(ROBOT_IP, franka.RealtimeConfig.kIgnore)
    except Exception as e:
        print(f"Could not connect to robot: {e}")
        sys.exit(-1)

    # Connect gripper if the trajectory includes gripper state
    gripper = None
    if any("gripper" in step for step in trajectory):
        try:
            gripper = franka.Gripper(ROBOT_IP)
        except Exception as e:
            print(f"Could not connect to gripper: {e}")
            sys.exit(-1)

    setDefaultBehaviour(robot)              # sets up baseline safety parameters
    move_robot_to_start_pose(robot, trajectory)

    # Apply the initial gripper state before starting the arm control loop
    if gripper is not None:
        initial_width = trajectory[0].get("gripper")
        if initial_width is not None:
            gripper.move(initial_width, kGripperMoveSpeed)


    # Configure the Asynchronous Safe Controller 
    joint_config = franka.AsyncPositionControlHandler.Configuration(
        maximum_joint_velocities=kDefaultMaximumVelocities,
        goal_tolerance=kDefaultGoalTolerance
    )
    result = franka.AsyncPositionControlHandler.configure(robot, joint_config)
    if result.error_message is not None:
        print(result.error_message); sys.exit(-1)
        
    position_control_handler = result.handler

    time_step = 0.020  # 50 Hz matching your planner file

    print("Pre-flight check passed. Starting execution in 3s... Hold the E-Stop!")
    time.sleep(3)

    last_gripper_width = trajectory[0].get("gripper") if gripper is not None else None

    try:
        for step_data in trajectory:
            if motion_finished:
                print("Stop requested. Halting hardware execution.")
                break

            loop_start = time.monotonic()

            target_feedback = position_control_handler.get_target_feedback()
            if target_feedback.error_message is not None:
                print(f"Error in feedback: {target_feedback.error_message}")
                sys.exit(-1)

            next_target = franka.AsyncPositionControlHandler.JointPositionTarget(
                joint_positions=step_data["joints"]
            )

            command_result = position_control_handler.set_joint_position_target(next_target)
            if command_result.error_message is not None:
                print(f"Hardware rejected target: {command_result.error_message}")
                sys.exit(-1)

            # Fire a non-blocking gripper command whenever the width changes
            if gripper is not None:
                new_width = step_data.get("gripper")
                if new_width is not None and new_width != last_gripper_width:
                    threading.Thread(
                        target=gripper.move,
                        args=(new_width, kGripperMoveSpeed),
                        daemon=True
                    ).start()
                    last_gripper_width = new_width

            sleep_time = time_step - (time.monotonic() - loop_start)
            if sleep_time > 0:
                time.sleep(sleep_time)
    finally:
        position_control_handler.stop_control()

    print("Execution complete.")

if __name__ == "__main__":
    run_hardware_execution()