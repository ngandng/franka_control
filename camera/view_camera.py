#!/usr/bin/env python3
import os
import sys

# ── SILENCE BACKGROUND WAYLAND/QT SPAM ──────────────────────────────────────
f_null = open(os.devnull, 'w')
old_stderr = sys.stderr
sys.stderr = f_null

import cv2
import numpy as np
import pyrealsense2 as rs

sys.stderr = old_stderr
f_null.close()
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # 1. Initialize RealSense Pipeline
    pipeline = rs.pipeline()
    config = rs.config()
    
    # Configure telemetry streams at standard 640x480 resolution
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    
    try:
        pipeline.start(config)
        print("✅ RealSense pipeline started successfully!")
    except Exception as e:
        print(f"❌ Failed to connect to RealSense camera: {e}")
        return

    # Create an alignment tool to align depth stream to color stream
    align = rs.align(rs.stream.color)

    print("\n🚀 Live stream active. Close the window or press 'q' to quit.")

    try:
        while True:
            # Gather frames and align them spatially
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()
            
            if not color_frame or not depth_frame:
                continue

            # Convert RealSense image frames to NumPy arrays
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())

            # Convert the raw depth map into a colored heatmap visualization
            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image, alpha=0.03), 
                cv2.COLORMAP_JET
            )

            # Stack both streams horizontally (side-by-side) to view both at once
            display_window = np.hstack((color_image, depth_colormap))

            # Display the side-by-side window
            cv2.imshow('RealSense Live Stream (RGB | Depth)', display_window)

            # Break loop instantly if user strikes the 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        # Stop hardware streams and clear UI windows cleanly on exit
        pipeline.stop()
        cv2.destroyAllWindows()
        print("👋 Camera stream closed cleanly.")

if __name__ == "__main__":
    main()