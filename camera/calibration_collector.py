import math
import time
import ast
from pathlib import Path
import numpy as np
import cv2
# from camera_utils import setup_camera_pipeline

import pylibfranka as franka


def read_calibration_waypoints_file(file_path="camera/calibration_waypoints.txt") -> list[list[float]]:
    """
        Read the file path and parse it into a list of joint configurations.
        Each line in the file should represent a joint configuration, e.g.:
        [0.0, -0.4, 0.0, -2.5, 0.0, 2.0, 0.0]
    """
    waypoints = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and line.startswith('"joint_angles":[') and line.endswith('],'):
                try:
                    waypoint = eval(line.split(":", 1)[1].strip().rstrip(','))  # Extract the list part
                    if isinstance(waypoint, list) and len(waypoint) == 7:
                        waypoints.append(waypoint)
                    else:
                        print(f"Invalid waypoint format (not a list of 7): {line}")
                except Exception as e:
                    print(f"Error parsing line: {line} | Error: {e}")

    # print(f"Successfully read {len(waypoints)} calibration waypoints from file.")
    # for i, wp in enumerate(waypoints):
    #     print(f"Waypoint {i+1}: {wp}")
    return waypoints



def move_robot_to_pose(robot, target_joints, controller, tolerance):
    
    # Placeholder for a real path planner. For now, we just return the target joints directly.
    # In a real implementation, you would generate intermediate waypoints here to ensure a safe trajectory.

    robot_state = robot.read_once()
    current_joints = list(robot_state.q)
    joint_errors = [abs(current - target) for current, target in zip(current_joints, target_joints)]
    max_joint_error = max(joint_errors)

    if max_joint_error <= tolerance:
        return

    print(
        "Robot is not at the target pose. "
        f"Moving to target pose first. Max joint error: {max_joint_error:.4f} rad "
        f"({math.degrees(max_joint_error):.2f} deg)."
    )


    # Interpolate slowly from current to target joints
    steps = 1000  # 20 seconds at 50Hz — slow and safe
    for i in range(steps):
        # Read feedback to check for errors
        target_feedback = controller.get_target_feedback()
        if target_feedback.error_message is not None:
            raise RuntimeError(f"Error in feedback during target pose move: {target_feedback.error_message}")

        # Interpolate each joint linearly towards the target pose
        loop_start = time.monotonic()
        alpha = i / max(steps - 1, 1)
        target = [c + alpha * (s - c) for c, s in zip(current_joints, target_joints)]
        next_target = franka.AsyncPositionControlHandler.JointPositionTarget(
            joint_positions=target
        )   # safety filters for the position control handler
        command_result = controller.set_joint_position_target(next_target)

        if command_result.error_message is not None:
            raise RuntimeError(f"Hardware rejected target: {command_result.error_message}")
        sleep_time = 0.020 - (time.monotonic() - loop_start)
        if sleep_time > 0:
            time.sleep(sleep_time)




def collect_calibration_data(camera_pipeline, camera_align, robot, controller, control_tolerance, chessboard_size=(7, 5)):

    file_path = "camera/calibration_waypoints.txt"
    waypoints = read_calibration_waypoints_file(file_path)
    if not waypoints:
        raise RuntimeError(f"No calibration waypoints found in {file_path}.")
    
    robot_poses = []
    images = []
    image_dir = Path("camera/calibration_images")
    image_dir.mkdir(parents=True, exist_ok=True)

    print("🏁 Starting automated calibration sweep...")
    
    for i, q_target in enumerate(waypoints):
        print(f"🤖 Moving to waypoint {i+1}/{len(waypoints)}...")
        move_robot_to_pose(robot, q_target, controller, control_tolerance)
        time.sleep(1.5) # Wait for the arm to settle completely so there's no motion blur
        
        # ========= Capture the stable camera frame =========
        frames = camera_pipeline.wait_for_frames()
        color_frame = camera_align.process(frames).get_color_frame()
        if not color_frame:
            continue  # Skip if no color frame is available
        
        color_image = np.asanyarray(color_frame.get_data())
        

        
        # ========== Capture robot pose ==========
        state = robot.read_once()
        T_b_g = np.array(state.O_T_EE).reshape((4,4), order='F')
        
        images.append(color_image)
        robot_poses.append(T_b_g)
        image_index = len(images)
        image_path = image_dir / f"{image_index}.png"
        if not cv2.imwrite(str(image_path), color_image):
            print(f" [!] Failed to save calibration image: {image_path}")
        else:
            print(f" [v] Saved calibration image: {image_path}")
        print(f" [v] Captured image and pose data for waypoint {i+1}")

    camera_pipeline.stop()
    return robot_poses, images
    # Pass 'robot_poses' and 'images' directly into cv2.calibrateHandEye() here!



if __name__ == "__main__":
    read_calibration_waypoints_file()