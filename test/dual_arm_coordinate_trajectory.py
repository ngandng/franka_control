import json
import time
import sys
import signal
import math
import threading

import pylibfranka as franka
from pylibfranka_examples.example_common import setDefaultBehaviour
from panda_joint_limits import START_MOVE_MAX_JOINT_VELOCITIES, TRAJECTORY_MAX_JOINT_VELOCITIES

# ============= GLOBAL CONFIGURATION =============
LEFT_ROBOT_IP = "172.16.0.3"
RIGHT_ROBOT_IP = "172.16.0.2"

GRIPPER_THRESHOLD = 0.002                       # 2mm buffer to ignore minor floating-point noise
home_q = [0, -0.5, 0, -2.5, 0, 2.0, 0.8]        # franka arm neutral pose

# Keep the hardware speed slightly above the generated trajectory limit
# (0.15 rad/s) so it can track without making a large catch-up movement.
kDefaultMaximumVelocities = TRAJECTORY_MAX_JOINT_VELOCITIES
# This must be smaller than the 10 ms trajectory increments.  A large goal
# tolerance lets the asynchronous controller accept several tiny targets at
# once, which appears as stop-go motion on the robot.
kDefaultGoalTolerance = 0.001                   # rad (about 0.06 degrees)
kStartJointTolerance = 0.01                     # rad (about 0.57 degrees)
kControlPeriod = 0.010                          # 100 Hz; must match generated JSON
kStartSettleTimeout = 5.0                       # seconds
kGripperMoveSpeed = 0.1                         # m/s
kGripperForce = 10.0                            # N
kGripperGraspEpsilonInner = 0.005               # m below requested grasp width
kGripperGraspEpsilonOuter = 0.005               # m above requested grasp width

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

        if not isinstance(step_data.get("time"), (int, float)):
            raise ValueError(f"Waypoint {index} must contain a numeric time value.")
        if index:
            period = step_data["time"] - trajectory[index - 1]["time"]
            if abs(period - kControlPeriod) > 1e-6:
                raise ValueError(
                    f"Waypoint {index} has a {period:.4f}s period, but this player runs at "
                    f"{kControlPeriod:.4f}s (100 Hz). Regenerate the trajectory before playback."
                )


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



def _send_joint_target(controller, joints, arm_label):
    result = controller.set_joint_position_target(
        franka.AsyncPositionControlHandler.JointPositionTarget(joint_positions=joints)
    )
    if result.error_message is not None:
        raise RuntimeError(f"{arm_label} arm rejected target: {result.error_message}")


def _check_controller_feedback(controller, arm_label):
    feedback = controller.get_target_feedback()
    if feedback.error_message is not None:
        raise RuntimeError(f"{arm_label} arm feedback error: {feedback.error_message}")


def _max_joint_error(robot, target_joints):
    current_joints = list(robot.read_once().q)
    return max(abs(current - target) for current, target in zip(current_joints, target_joints))


def _raise_gripper_error(gripper_errors):
    if gripper_errors:
        raise RuntimeError(gripper_errors.pop(0))


def _command_gripper_worker(gripper, width, previous_width, arm_label, gripper_errors):
    """Run a blocking gripper RPC outside the 100 Hz arm command loop."""
    try:
        if width < previous_width:
            # Closing is force-controlled: keep closing until the requested
            # width is reached or an object is contacted with this force.
            succeeded = gripper.grasp(
                width,
                kGripperMoveSpeed,
                kGripperForce,
                kGripperGraspEpsilonInner,
                kGripperGraspEpsilonOuter,
            )
            action = "force grasp"
        else:
            # Opening does not need force control.
            succeeded = gripper.move(width, kGripperMoveSpeed)
            action = "position move"
        if not succeeded:
            try:
                final_width = gripper.read_once().width
                final_state = (
                    f" Measured final opening: {final_width:.3f} m; expected "
                    f"{width - kGripperGraspEpsilonInner:.3f}–"
                    f"{width + kGripperGraspEpsilonOuter:.3f} m."
                )
            except Exception as state_error:
                final_state = f" Could not read final gripper state: {state_error}"
            gripper_errors.append(
                f"{arm_label} gripper rejected {action} to {width:.3f} m."
                f"{final_state}"
            )
    except Exception as exc:
        gripper_errors.append(
            f"{arm_label} gripper command to {width:.3f} m failed: {exc}"
        )


def _start_gripper_command(
    gripper,
    width,
    previous_width,
    arm_label,
    gripper_workers,
    gripper_errors,
):
    previous_worker = gripper_workers.get(arm_label)
    if previous_worker is not None and previous_worker.is_alive():
        raise RuntimeError(
            f"{arm_label} gripper received a new command before its previous command completed."
        )

    worker = threading.Thread(
        target=_command_gripper_worker,
        args=(gripper, width, previous_width, arm_label, gripper_errors),
        daemon=True,
    )
    gripper_workers[arm_label] = worker
    worker.start()


def move_robots_to_start_pose(
    left_robot,
    right_robot,
    left_start,
    right_start,
    left_controller,
    right_controller,
    tolerance=kStartJointTolerance,
):
    """Move both arms from one snapshot to their start targets on the same tick."""
    left_current = list(left_robot.read_once().q)
    right_current = list(right_robot.read_once().q)
    max_initial_error = max(
        max(abs(current - target) for current, target in zip(left_current, left_start)),
        max(abs(current - target) for current, target in zip(right_current, right_start)),
    )

    # At least 20 s; extend it when needed to respect the start velocity cap.
    steps = max(
        1001,
        max(
            int(math.ceil(abs(target - current) / (velocity * kControlPeriod))) + 1
            for current_arm, target_arm in ((left_current, left_start), (right_current, right_start))
            for current, target, velocity in zip(
                current_arm, target_arm, START_MOVE_MAX_JOINT_VELOCITIES
            )
        ),
    )
    print(
        "Moving both robots to the trajectory start pose together "
        f"({(steps - 1) * kControlPeriod:.1f}s maximum, "
        f"initial max error {math.degrees(max_initial_error):.2f} deg)."
    )

    for index in range(steps):
        if motion_finished:
            return

        loop_start = time.monotonic()
        _check_controller_feedback(left_controller, "Left")
        _check_controller_feedback(right_controller, "Right")

        alpha = index / (steps - 1)
        left_target = [current + alpha * (target - current) for current, target in zip(left_current, left_start)]
        right_target = [current + alpha * (target - current) for current, target in zip(right_current, right_start)]

        # These calls are back-to-back in the same 50 Hz control tick.  Do not
        # use separate Python threads: a shared tick makes coordination and
        # failure handling deterministic.
        _send_joint_target(left_controller, left_target, "Left")
        _send_joint_target(right_controller, right_target, "Right")

        sleep_time = kControlPeriod - (time.monotonic() - loop_start)
        if sleep_time > 0:
            time.sleep(sleep_time)

    # Sending the final target is not evidence that the hardware reached it.
    # Hold it and read both measured states until they are within tolerance.
    settle_deadline = time.monotonic() + kStartSettleTimeout
    while time.monotonic() < settle_deadline:
        left_error = _max_joint_error(left_robot, left_start)
        right_error = _max_joint_error(right_robot, right_start)
        if left_error <= tolerance and right_error <= tolerance:
            print(
                "Both robots reached the start pose "
                f"(left {math.degrees(left_error):.2f} deg, "
                f"right {math.degrees(right_error):.2f} deg maximum error)."
            )
            return

        _send_joint_target(left_controller, left_start, "Left")
        _send_joint_target(right_controller, right_start, "Right")
        time.sleep(kControlPeriod)

    assert_robot_is_at_start(left_robot, left_start, "Left", tolerance=tolerance)
    assert_robot_is_at_start(right_robot, right_start, "Right", tolerance=tolerance)
    raise RuntimeError("Both robots did not settle at the trajectory start pose before timeout.")


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
    gripper_workers = {}
    gripper_errors = []
    try:
        left_position_control_handler = configure_position_controller(left_robot)
        right_position_control_handler = configure_position_controller(right_robot)

        left_start = trajectory[0]["left_joints"]
        right_start = trajectory[0]["right_joints"]

        move_robots_to_start_pose(
            left_robot,
            right_robot,
            left_start,
            right_start,
            left_position_control_handler,
            right_position_control_handler,
        )
        time.sleep(0.5)

        time_step = kControlPeriod  # 100 Hz matching trajectory file

        print("Pre-flight check passed. Starting execution in 3s... Hold the E-Stop!")
        time.sleep(3)

        last_left_gripper = trajectory[0].get("left_gripper")
        last_right_gripper = trajectory[0].get("right_gripper")

        next_tick = time.monotonic()
        late_tick_count = 0
        max_lateness = 0.0

        for step_index, step_data in enumerate(trajectory):
            if motion_finished:
                print("Stop requested. Halting hardware execution.")
                break

            _raise_gripper_error(gripper_errors)

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
                        _start_gripper_command(
                            left_gripper,
                            new_left_width,
                            last_left_gripper,
                            "Left",
                            gripper_workers,
                            gripper_errors,
                        )
                        last_left_gripper = new_left_width

            if right_gripper is not None:
                new_right_width = step_data.get("right_gripper")
                if new_right_width is not None and last_right_gripper is not None:
                    if abs(new_right_width - last_right_gripper) > GRIPPER_THRESHOLD:
                        print(
                            f"Right gripper action detected ({last_right_gripper}m -> {new_right_width}m)."
                        )
                        _start_gripper_command(
                            right_gripper,
                            new_right_width,
                            last_right_gripper,
                            "Right",
                            gripper_workers,
                            gripper_errors,
                        )
                        last_right_gripper = new_right_width

            # Keep an absolute schedule while the loop is on time.  If a tick
            # overruns, reset the following deadline rather than sending later
            # targets back-to-back to catch up; preserving target spacing is
            # smoother and safer than preserving wall-clock completion time.
            next_tick += time_step
            sleep_time = next_tick - time.monotonic()
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                lateness = -sleep_time
                late_tick_count += 1
                max_lateness = max(max_lateness, lateness)
                if late_tick_count <= 3:
                    print(
                        f"Playback tick {step_index} was {lateness * 1000:.1f} ms late; "
                        "holding trajectory spacing instead of catching up."
                    )
                next_tick = time.monotonic()

        _raise_gripper_error(gripper_errors)
        if late_tick_count:
            print(
                f"Playback timing: {late_tick_count} late ticks; "
                f"worst lateness {max_lateness * 1000:.1f} ms."
            )

        if not motion_finished:
            print("Trajectory playback finished. Waiting for user to exit...")
            while not motion_finished:
                time.sleep(0.1)

    finally:
        for worker in gripper_workers.values():
            worker.join(timeout=1.0)
        _raise_gripper_error(gripper_errors)
        if left_position_control_handler is not None:
            left_position_control_handler.stop_control()
        if right_position_control_handler is not None:
            right_position_control_handler.stop_control()

    print("Execution complete.")


if __name__ == "__main__":
    run_hardware_execution(filename="path_data/dual_arm_trajectory.json")
