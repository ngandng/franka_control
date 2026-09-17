# Copyright (c) 2026 Franka Robotics GmbH
# Apache-2.0

# This example demonstrates asynchronous position control of a Franka robot using the
# pylibfranka library. It connects to the robot, sets up an asynchronous position control
# handler, and continuously updates joint position targets in a loop until interrupted, leveraging
# the latest low-rate control API.

import signal
import sys
import time
import math
import argparse
from datetime import timedelta

import pylibfranka as franka
from pylibfranka_examples.example_common import setDefaultBehaviour

kDefaultMaximumVelocities = [0.655, 0.655, 0.655, 0.655, 1.315, 1.315, 1.315]
kDefaultGoalTolerance = 10.0
kStartJointTolerance = 0.05

home_q = [0, -0.5, 0, -2.5, 0, 2.0, 0.8]        # franka arm neutral pose

motion_finished = False


def signal_handler(sig, frame):
    global motion_finished
    if sig == signal.SIGINT:
        motion_finished = True

def move_robot_to_start_pose(robot, controller, tolerance=kStartJointTolerance):

    robot_state = robot.read_once()
    current_joints = list(robot_state.q)
    joint_errors = [abs(current - target) for current, target in zip(current_joints, home_q)]
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
        target = [c + alpha * (s - c) for c, s in zip(current_joints, home_q)]
        next_target = franka.AsyncPositionControlHandler.JointPositionTarget(
            joint_positions=target
        )   # safety filters for the position control handler
        command_result = controller.set_joint_position_target(next_target)

        if command_result.error_message is not None:
            raise RuntimeError(f"Hardware rejected target: {command_result.error_message}")
        sleep_time = 0.020 - (time.monotonic() - loop_start)
        if sleep_time > 0:
            time.sleep(sleep_time)

    assert_robot_is_at_start(robot, home_q, tolerance=tolerance)


def assert_robot_is_at_start(robot, start_joints, tolerance=kStartJointTolerance):
    robot_state = robot.read_once()
    current_joints = list(robot_state.q)
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



def main():

    signal.signal(signal.SIGINT, signal_handler)

    left_ip = "172.16.0.3"
    right_ip = "172.16.0.2"

    try:
        left_robot = franka.Robot(left_ip, franka.RealtimeConfig.kIgnore)
        right_robot = franka.Robot(right_ip, franka.RealtimeConfig.kIgnore)
    except Exception as e:
        print(f"Could not connect to robot: {e}")
        sys.exit(-1)

    setDefaultBehaviour(left_robot)              # sets up baseline safety parameters
    setDefaultBehaviour(right_robot)              # sets up baseline safety parameters

    initial_position = [0,
                        -math.pi / 4,
                        0,
                        -3 * math.pi / 4,
                        0,
                        math.pi / 2,
                        math.pi / 4]        # 7 movable joints

    time_elapsed = 0.0
    direction = 1.0
    time_since_last_log = 0.0

    def calculate_joint_position_target(period_sec):
        nonlocal time_elapsed, direction, time_since_last_log

        time_elapsed += period_sec

        # TARGET POSE: interpolate of 7 points
        target_positions = [
            initial_position[i] + direction * 0.25
            for i in range(7)
        ]

        time_since_last_log += period_sec
        if time_since_last_log >= 1.0:      # Go back and forth every 1 second
            direction *= -1.0
            time_since_last_log = 0.0

        return franka.AsyncPositionControlHandler.JointPositionTarget(
            joint_positions=target_positions
        )


    joint_position_control_configuration = \
        franka.AsyncPositionControlHandler.Configuration(
            maximum_joint_velocities=kDefaultMaximumVelocities,
            goal_tolerance=kDefaultGoalTolerance
        )   # safety filters for the position control handler

    result = franka.AsyncPositionControlHandler.configure(
        left_robot,
        joint_position_control_configuration
    )

    if result.error_message is not None:
        print(result.error_message)
        sys.exit(-1)

    left_position_control_handler = result.handler

    result = franka.AsyncPositionControlHandler.configure(
        right_robot,
        joint_position_control_configuration
    )

    if result.error_message is not None:
        print(result.error_message)
        sys.exit(-1)

    right_position_control_handler = result.handler
    left_target_feedback = left_position_control_handler.get_target_feedback()
    right_target_feedback = right_position_control_handler.get_target_feedback()


    ### CONTROL LOOP ###
    time_step = 0.020  # 20 ms, 50 Hz

    global motion_finished
    while not motion_finished:
        loop_start = time.monotonic()

        # 1. Check for hardware errors
        left_target_feedback = left_position_control_handler.get_target_feedback()
        if left_target_feedback.error_message is not None:
            print(left_target_feedback.error_message)
            sys.exit(-1)
        right_target_feedback = right_position_control_handler.get_target_feedback()
        if right_target_feedback.error_message is not None:
            print(right_target_feedback.error_message)
            sys.exit(-1)

        # 2. Calculate and push the next goal state
        next_target = calculate_joint_position_target(time_step)
        left_command_result = left_position_control_handler.set_joint_position_target(next_target)
        right_command_result = right_position_control_handler.set_joint_position_target(next_target)

        if left_command_result.error_message is not None:
            print(left_command_result.error_message)
            sys.exit(-1)
        if right_command_result.error_message is not None:
            print(right_command_result.error_message)
            sys.exit(-1)

        # 3. Check if the motion duration has been reached
        if time_elapsed > 10.0:
            left_position_control_handler.stop_control()
            right_position_control_handler.stop_control()
            motion_finished = True
            print("Control finished")
            break
        
        # 4. Sleep to maintain the control loop rate
        sleep_time = time_step - (time.monotonic() - loop_start)
        if sleep_time > 0:
            time.sleep(sleep_time)


if __name__ == "__main__":
    main()