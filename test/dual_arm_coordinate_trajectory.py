import json
import time
import sys
import signal
import math

import pylibfranka as franka
from pylibfranka_examples.example_common import setDefaultBehaviour

# ============= GLOBAL CONFIGURATION =============
LEFT_ROBOT_IP = "172.16.0.3"
RIGHT_ROBOT_IP = "172.16.0.2"

GRIPPER_THRESHOLD = 0.002                       # 2mm buffer to ignore minor floating-point noise
home_q = [0, -0.5, 0, -2.5, 0, 2.0, 0.8]        # franka arm neutral pose

kDefaultMaximumVelocities = [0.655, 0.655, 0.655, 0.655, 1.315, 1.315, 1.315]
kDefaultGoalTolerance = 10.0
kStartJointTolerance = 0.05
kGripperMoveSpeed = 0.1                         # m/s
kGripperForce = 60.0                            # N

motion_finished = False
#=================================================



def signal_handler(sig, frame):
    global motion_finished
    if sig == signal.SIGINT:
        motion_finished = True


def validate_trajectory(trajectory):
    if not isinstance(trajectory, list) or not trajectory:
        raise ValueError("Trajectory must be a non-empty list of waypoints.")

    for index, step_data in enumerate(trajectory):
        left_joints = step_data.get("left_joints")
        right_joints = step_data.get("right_joints")
        if left_joints is None or right_joints is None:
            raise ValueError(
                f"Waypoint {index} must contain both left_joints and right_joints for dual-arm playback."
            )
        if not isinstance(left_joints, list) or len(left_joints) != 7:
            raise ValueError(f"Waypoint {index} must contain 7 left joint values.")
        if not isinstance(right_joints, list) or len(right_joints) != 7:
            raise ValueError(f"Waypoint {index} must contain 7 right joint values.")


def assert_robot_is_at_start(robot, start_joints, arm_label, tolerance=kStartJointTolerance):
    robot_state = robot.read_once()
    current_joints = list(robot_state.q)
    joint_errors = [abs(current - target) for current, target in zip(current_joints, start_joints)]
    max_joint_error = max(joint_errors)

    if max_joint_error > tolerance:
        current_deg = [round(math.degrees(value), 2) for value in current_joints]
        start_deg = [round(math.degrees(value), 2) for value in start_joints]
        error_deg = [round(math.degrees(value), 2) for value in joint_errors]
        raise RuntimeError(
            f"{arm_label} robot is not at the trajectory start pose. "
            f"Max joint error is {max_joint_error:.4f} rad ({math.degrees(max_joint_error):.2f} deg), "
            f"which exceeds the tolerance of {tolerance:.4f} rad ({math.degrees(tolerance):.2f} deg).\n"
            f"Current joints (deg): {current_deg}\n"
            f"Start joints (deg):   {start_deg}\n"
            f"Absolute error (deg): {error_deg}\n"
            "Move the arm to the start pose before replaying this file."
        )



def move_robot_to_start_pose(robot, start_joints, controller, arm_label, tolerance=kStartJointTolerance):
    robot_state = robot.read_once()
    current_joints = list(robot_state.q)
    joint_errors = [abs(current - target) for current, target in zip(current_joints, start_joints)]
    max_joint_error = max(joint_errors)

    if max_joint_error <= tolerance:
        return

    print(
        f"{arm_label} robot is not at the trajectory start pose. "
        f"Moving to start pose first. Max joint error: {max_joint_error:.4f} rad "
        f"({math.degrees(max_joint_error):.2f} deg)."
    )


    # Interpolate slowly from current to start joints
    steps = 1000  # 20 seconds at 50Hz — slow and safe
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

    assert_robot_is_at_start(robot, start_joints, arm_label, tolerance=tolerance)


def configure_position_controller(robot):
    joint_position_control_configuration = franka.AsyncPositionControlHandler.Configuration(
        maximum_joint_velocities=kDefaultMaximumVelocities,
        goal_tolerance=kDefaultGoalTolerance,
    )
    result = franka.AsyncPositionControlHandler.configure(
        robot,
        joint_position_control_configuration,
    )
    if result.error_message is not None:
        raise RuntimeError(result.error_message)
    return result.handler



def run_hardware_execution(filename="path_data/dual_arm_trajectory.json"):

    #=========== LOAD THE FILE =================
    with open(filename, 'r') as f:
        trajectory = json.load(f)

    validate_trajectory(trajectory)
    # ===== SETUP ROBOT CONFIGURATION AND SAFETY =======
    signal.signal(signal.SIGINT, signal_handler)

    try:
        left_robot = franka.Robot(LEFT_ROBOT_IP, franka.RealtimeConfig.kIgnore)
        right_robot = franka.Robot(RIGHT_ROBOT_IP, franka.RealtimeConfig.kIgnore)
    except Exception as e:
        print(f"Could not connect to robots: {e}")
        sys.exit(-1)

    left_gripper = None
    right_gripper = None
    try:
        left_gripper = franka.Gripper(LEFT_ROBOT_IP)
        left_gripper.homing()
    except Exception as e:
        print(f"Could not connect to left gripper: {e}")

    try:
        right_gripper = franka.Gripper(RIGHT_ROBOT_IP)
        right_gripper.homing()
    except Exception as e:
        print(f"Could not connect to right gripper: {e}")

    setDefaultBehaviour(left_robot)
    setDefaultBehaviour(right_robot)

    # ========== TRAJECTORY EXECUTION ================
    left_position_control_handler = None
    right_position_control_handler = None
    try:
        left_position_control_handler = configure_position_controller(left_robot)
        right_position_control_handler = configure_position_controller(right_robot)

        left_start = trajectory[0]["left_joints"]
        right_start = trajectory[0]["right_joints"]

        move_robot_to_start_pose(
            left_robot,
            left_start,
            left_position_control_handler,
            arm_label="Left",
        )
        move_robot_to_start_pose(
            right_robot,
            right_start,
            right_position_control_handler,
            arm_label="Right",
        )
        time.sleep(0.5)

        time_step = 0.020  # 50 Hz matching trajectory file

        print("Pre-flight check passed. Starting execution in 3s... Hold the E-Stop!")
        time.sleep(3)

        last_left_gripper = trajectory[0].get("left_gripper")
        last_right_gripper = trajectory[0].get("right_gripper")

        for step_data in trajectory:
            if motion_finished:
                print("Stop requested. Halting hardware execution.")
                break

            loop_start = time.monotonic()

            left_feedback = left_position_control_handler.get_target_feedback()
            if left_feedback.error_message is not None:
                print(f"Left arm feedback error: {left_feedback.error_message}")
                sys.exit(-1)

            right_feedback = right_position_control_handler.get_target_feedback()
            if right_feedback.error_message is not None:
                print(f"Right arm feedback error: {right_feedback.error_message}")
                sys.exit(-1)

            left_joints = step_data["left_joints"]
            right_joints = step_data["right_joints"]

            left_result = left_position_control_handler.set_joint_position_target(
                franka.AsyncPositionControlHandler.JointPositionTarget(joint_positions=left_joints)
            )
            if left_result.error_message is not None:
                print(f"Left arm rejected target: {left_result.error_message}")
                sys.exit(-1)

            right_result = right_position_control_handler.set_joint_position_target(
                franka.AsyncPositionControlHandler.JointPositionTarget(joint_positions=right_joints)
            )
            if right_result.error_message is not None:
                print(f"Right arm rejected target: {right_result.error_message}")
                sys.exit(-1)

            if left_gripper is not None:
                new_left_width = step_data.get("left_gripper")
                if new_left_width is not None and last_left_gripper is not None:
                    if abs(new_left_width - last_left_gripper) > GRIPPER_THRESHOLD:
                        print(
                            f"Left gripper action detected ({last_left_gripper}m -> {new_left_width}m)."
                        )
                        left_gripper.move(new_left_width, kGripperMoveSpeed)
                        last_left_gripper = new_left_width

            if right_gripper is not None:
                new_right_width = step_data.get("right_gripper")
                if new_right_width is not None and last_right_gripper is not None:
                    if abs(new_right_width - last_right_gripper) > GRIPPER_THRESHOLD:
                        print(
                            f"Right gripper action detected ({last_right_gripper}m -> {new_right_width}m)."
                        )
                        right_gripper.move(new_right_width, kGripperMoveSpeed)
                        last_right_gripper = new_right_width


            # ====== TIMING REGULATION ======
            sleep_time = time_step - (time.monotonic() - loop_start)
            if sleep_time > 0:
                time.sleep(sleep_time)

        if not motion_finished:
            print("Trajectory playback finished. Waiting for user to exit...")
            while not motion_finished:
                time.sleep(0.1)

    finally:
        if left_position_control_handler is not None:
            left_position_control_handler.stop_control()
        if right_position_control_handler is not None:
            right_position_control_handler.stop_control()

    print("Execution complete.")


if __name__ == "__main__":
    run_hardware_execution(filename="path_data/dual_arm_trajectory.json")