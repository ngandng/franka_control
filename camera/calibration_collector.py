import time
import numpy as np
import cv2
from camera_utils import setup_camera_pipeline

# 1. Define 5 to 6 distinct joint configurations that look at your checkerboard from different angles
# Shift joint 6 and 7 slightly between poses to get a good calibration variance
CALIBRATION_WAYPOINTS = [
    [0.0, -0.4, 0.0, -2.5, 0.0, 2.0, 0.0],   # Pose 1: Straight down
    [0.1, -0.3, 0.1, -2.4, 0.2, 1.9, 0.3],   # Pose 2: Tilted left
    [-0.1, -0.5, -0.1, -2.6, -0.2, 2.1, -0.3], # Pose 3: Tilted right
    # ... add a few more variations here
]

def main():
    pipeline, align, intrinsics = setup_camera_pipeline()
    # robot = AsyncPositionControlHandler("172.16.0.2") # Your robot handler
    
    robot_poses = []
    images = []

    print("🏁 Starting automated calibration sweep...")
    
    for i, q_target in enumerate(CALIBRATION_WAYPOINTS):
        print(f"🤖 Moving to waypoint {i+1}/{len(CALIBRATION_WAYPOINTS)}...")
        # robot.execute_joint_trajectory_blocking(q_target)
        time.sleep(1.5) # Wait for the arm to settle completely so there's no motion blur
        
        # Capture the stable camera frame
        frames = pipeline.wait_for_frames()
        color_frame = align.process(frames).get_color_frame()
        img = np.asanyarray(color_frame.get_data())
        
        # Read the exact live O_T_EE matrix from libfranka
        # state = robot.get_state()
        # T_b_g = np.array(state.O_T_EE).reshape((4,4), order='F')
        
        images.append(img)
        # robot_poses.append(T_b_g)
        print(f"📸 Captured image and pose data for waypoint {i+1}")

    pipeline.stop()
    
    # Pass 'robot_poses' and 'images' directly into cv2.calibrateHandEye() here!