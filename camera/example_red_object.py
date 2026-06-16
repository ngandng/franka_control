
import os
import sys

import pyrealsense2 as rs
import numpy as np
import cv2


# Mute the Qt font warning logs completely
os.environ["QT_LOGGING_RULES"] = "qt.qpa.fonts=false"

# Suppress Wayland platform warnings from cluttering the backend
os.environ["XDG_SESSION_TYPE"] = "x11"


def test_camera_connection():
    try:
        # Start the camera pipeline
        pipeline = rs.pipeline()
        pipeline.start()

        # Get device information
        device = pipeline.get_active_profile().get_device()
        print(f"Success! Connected to: {device.get_info(rs.camera_info.name)}")
        print(f"Serial Number: {device.get_info(rs.camera_info.serial_number)}")

        pipeline.stop()
    except Exception as e:
        print(f"Error: {e}. Is the D435 plugged into a USB 3.0 port?")


def main():
    # 1. Setup RealSense Pipeline
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    profile = pipeline.start(config)
    
    intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    align = rs.align(rs.stream.color)

    # 2. Define Color Thresholds (Example: Bright Red Block)
    # Note: Hue ranges from 0-180 in OpenCV. 
    # Red sits at the very edge of the spectrum, so it can wrap around 0 and 180.
    lower_color = np.array([0, 120, 70])      # Lower [Hue, Saturation, Value]
    upper_color = np.array([10, 255, 255])    # Upper [Hue, Saturation, Value]

    try:
        print("Lightweight color tracking started. Press 'q' to exit.")
        while True:
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()
            
            if not color_frame or not depth_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())
            
            # 3. Convert image to HSV space
            hsv_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
            
            # 4. Filter out everything except our target color range
            mask = cv2.inRange(hsv_image, lower_color, upper_color)
            
            # Optional: Clean up minor image noise using morphology operations
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

            # 5. Find the outlines (contours) of the colored object
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Find the largest colored block in view to ignore background spots
                largest_contour = max(contours, key=cv2.contourArea)
                
                # Check if the object is large enough to be real (filter out tiny pixels)
                if cv2.contourArea(largest_contour) > 400:
                    # Calculate spatial image moments to find the center pixel
                    M = cv2.moments(largest_contour)
                    if M["m00"] != 0:
                        pixel_x = int(M["m10"] / M["m00"])
                        pixel_y = int(M["m01"] / M["m00"])
                        
                        # 6. Extract Depth and Project to 3D Camera Coordinates
                        distance = depth_frame.get_distance(pixel_x, pixel_y)
                        
                        if distance > 0:
                            # Turn pixels into real-world [X, Y, Z] metrics in meters
                            camera_xyz = rs.rs2_deproject_pixel_to_point(intr, [pixel_x, pixel_y], distance)
                            x_c, y_c, z_c = camera_xyz
                            
                            # Draw visual indicators for verification
                            cv2.drawContours(color_image, [largest_contour], -1, (0, 255, 0), 2)
                            cv2.circle(color_image, (pixel_x, pixel_y), 5, (0, 0, 255), -1)
                            cv2.putText(color_image, f"XYZ: {x_c:.2f}, {y_c:.2f}, {z_c:.2f}m", 
                                        (pixel_x + 10, pixel_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Show the regular camera view and the binary filter side-by-side
            cv2.imshow('Camera Stream', color_image)
            cv2.imshow('Color Mask Filter', mask)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    test_camera_connection()
    main()