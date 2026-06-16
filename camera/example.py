import pyrealsense2 as rs
import numpy as np
import cv2

def main():
    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Start streaming
    profile = pipeline.start(config)
    
    # Get camera intrinsic matrix parameters (vital for 3D projection math)
    intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()

    # Create an align object to perfectly match depth pixels to RGB pixels
    align = rs.align(rs.stream.color)

    try:
        print("Camera ready. Looking for objects... Press 'q' to quit.")
        while True:
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()
            
            if not color_frame or not depth_frame:
                continue

            # Convert images to numpy arrays
            color_image = np.asanyarray(color_frame.get_data())
            
            # 💡 PLACEHOLDER FOR YOUR DETECTION MODEL (YOLO, OpenCV contours, etc.)
            # Let's pretend your model detected an object box and found the center pixel:
            pixel_x, pixel_y = 320, 240  # Exact center of a 640x480 frame
            
            # Query the exact distance to that pixel in meters
            distance = depth_frame.get_distance(pixel_x, pixel_y)

            if distance > 0:
                # Deproject pixel coordinates into real 3D Camera Space coordinates
                # This uses camera focal length and center principal points to calculate true metrics
                camera_coordinate = rs.rs2_deproject_pixel_to_point(intr, [pixel_x, pixel_y], distance)
                
                # camera_coordinate is an array: [X_meters, Y_meters, Z_meters]
                x_c, y_c, z_c = camera_coordinate
                
                # Draw crosshair and telemetry data on screen
                cv2.circle(color_image, (pixel_x, pixel_y), 5, (0, 0, 255), -1)
                cv2.putText(color_image, f"XYZ: {x_c:.2f}, {y_c:.2f}, {z_c:.2f}m", 
                            (pixel_x + 10, pixel_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            cv2.imshow('RealSense Tracking Stream', color_image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()