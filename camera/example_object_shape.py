#!/usr/bin/env python3
import os
import sys

# ── SILENCE BACKGROUND OCV/QT SPAM ──────────────────────────────────────────
f_null = open(os.devnull, 'w')
old_stderr = sys.stderr
sys.stderr = f_null

import cv2
import numpy as np
import pyrealsense2 as rs

sys.stderr = old_stderr
f_null.close()
# ─────────────────────────────────────────────────────────────────────────────

def get_color_ranges():
    return {
        "Red": (np.array([0, 120, 70]), np.array([10, 255, 255])),
        "Blue": (np.array([95, 80, 50]), np.array([130, 255, 255])),
        "Green": (np.array([35, 60, 50]), np.array([85, 255, 255])),
        "Yellow": (np.array([20, 100, 80]), np.array([35, 255, 255])),
    }


def setup_camera_pipeline():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    profile = pipeline.start(config)
    color_stream = profile.get_stream(rs.stream.color)
    intrinsics = color_stream.as_video_stream_profile().get_intrinsics()
    align = rs.align(rs.stream.color)
    return pipeline, align, intrinsics


def classify_shape(contour):
    perimeter = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
    num_corners = len(approx)

    if num_corners == 3:
        return "Triangle"
    if num_corners == 4:
        _, _, w, h = cv2.boundingRect(approx)
        aspect_ratio = w / float(h)
        return "Square" if 0.9 <= aspect_ratio <= 1.1 else "Rectangle"
    return "Cylinder/Circle"


def extract_pose(depth_frame, intrinsics, contour):
    rect = cv2.minAreaRect(contour)
    (box_cx, box_cy), (box_w, box_h), yaw_angle = rect

    if box_w < box_h:
        yaw_angle += 90.0

    pixel_x, pixel_y = int(box_cx), int(box_cy)
    z_depth = depth_frame.get_distance(pixel_x, pixel_y)
    if z_depth <= 0:
        return None

    x_c, y_c, z_c = rs.rs2_deproject_pixel_to_point(intrinsics, [pixel_x, pixel_y], z_depth)
    return {
        "rect": rect,
        "pixel": (pixel_x, pixel_y),
        "xyz": (x_c, y_c, z_c),
        "yaw": yaw_angle,
    }


def draw_detection(color_image, color_name, shape, pose):
    rect = pose["rect"]
    pixel_x, pixel_y = pose["pixel"]
    x_c, y_c, z_c = pose["xyz"]
    yaw_angle = pose["yaw"]

    box_points = cv2.boxPoints(rect)
    box_points = np.intp(box_points)
    cv2.drawContours(color_image, [box_points], 0, (0, 255, 0), 2)
    cv2.circle(color_image, (pixel_x, pixel_y), 5, (0, 0, 255), -1)

    label = f"{color_name} {shape} | Yaw: {yaw_angle:.1f}deg"
    coord_label = f"XYZ: [{x_c:.3f}, {y_c:.3f}, {z_c:.3f}]m"

    cv2.putText(color_image, label, (pixel_x - 40, pixel_y - 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(color_image, coord_label, (pixel_x - 60, pixel_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)

    print(f"Identified: {color_name:<6} {shape:<9} | Pos: [{x_c:+.3f}, {y_c:+.3f}, {z_c:.3f}]m | Yaw: {yaw_angle:+.1f}°")


def process_frame(color_image, depth_frame, intrinsics, color_ranges):
    hsv_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
    kernel = np.ones((5, 5), np.uint8)
    combined_mask = np.zeros(hsv_image.shape[:2], dtype=np.uint8)

    for color_name, (lower_color, upper_color) in color_ranges.items():
        mask = cv2.inRange(hsv_image, lower_color, upper_color)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        combined_mask = cv2.bitwise_or(combined_mask, mask)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) < 500:
                continue

            shape = classify_shape(contour)
            pose = extract_pose(depth_frame, intrinsics, contour)
            if pose is None:
                continue

            draw_detection(color_image, color_name, shape, pose)

    return combined_mask


def stream_loop(pipeline, align, intrinsics, color_ranges):
    while True:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)

        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        if not color_frame or not depth_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())
        combined_mask = process_frame(color_image, depth_frame, intrinsics, color_ranges)

        cv2.imshow('RealSense Workspace Stream', color_image)
        cv2.imshow('Active Color Mask', combined_mask)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


def main():
    try:
        pipeline, align, intrinsics = setup_camera_pipeline()
    except Exception as e:
        print(f"Failed to connect to RealSense camera: {e}")
        return
    color_ranges = get_color_ranges()

    print("\n Camera pipeline active. Looking for objects...")
    print("Press 'q' in the graphics window to exit safely.\n")

    try:
        stream_loop(pipeline, align, intrinsics, color_ranges)

    finally:
        # Stop streams and clear windows cleanly on exit
        pipeline.stop()
        cv2.destroyAllWindows()
        print("\n👋 Camera stream terminated cleanly.")

if __name__ == "__main__":
    main()