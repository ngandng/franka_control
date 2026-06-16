# camera_module.py
import cv2
import numpy as np
import pyrealsense2 as rs


class RealSenseTracker:
    def __init__(self, lower_hsv=None, upper_hsv=None, color_ranges=None):
        self.pipeline = None
        self.align = None
        self.intrinsics = None
        self.min_area = 500

        self.T_flange_to_camera = np.array([
            [1.0,  0.0,  0.0,  0.05],   # X-offset (5 cm forward)
            [0.0,  1.0,  0.0,  0.00],   # Y-offset
            [0.0,  0.0,  1.0,  0.10],   # Z-offset (10 cm above tool flange)
            [0.0,  0.0,  0.0,  1.00]
        ])

        if color_ranges is not None:
            self.color_ranges = color_ranges
        elif lower_hsv is not None and upper_hsv is not None:
            self.color_ranges = {
                "Target": (np.array(lower_hsv), np.array(upper_hsv)),
            }
        else:
            self.color_ranges = self.get_color_ranges()

        self.setup_camera_pipeline()

    def get_color_ranges(self):
        return {
            "Red": (np.array([0, 120, 70]), np.array([10, 255, 255])),
            "Blue": (np.array([95, 80, 50]), np.array([130, 255, 255])),
            "Green": (np.array([35, 60, 50]), np.array([85, 255, 255])),
            "Yellow": (np.array([20, 100, 80]), np.array([35, 255, 255])),
        }

    def setup_camera_pipeline(self):
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        profile = self.pipeline.start(config)
        color_stream = profile.get_stream(rs.stream.color)
        self.intrinsics = color_stream.as_video_stream_profile().get_intrinsics()
        self.align = rs.align(rs.stream.color)

    def classify_shape(self, contour):
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

    def extract_pose(self, depth_frame, contour):
        rect = cv2.minAreaRect(contour)
        (box_cx, box_cy), (box_w, box_h), yaw_angle = rect

        if box_w < box_h:
            yaw_angle += 90.0

        pixel_x, pixel_y = int(box_cx), int(box_cy)
        z_depth = depth_frame.get_distance(pixel_x, pixel_y)
        if z_depth <= 0:
            return None

        x_c, y_c, z_c = rs.rs2_deproject_pixel_to_point(self.intrinsics, [pixel_x, pixel_y], z_depth)
        return {
            "rect": rect,
            "pixel": (pixel_x, pixel_y),
            "xyz": (x_c, y_c, z_c),
            "yaw": yaw_angle,
        }

    def draw_detection(self, color_image, color_name, shape, pose):
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

        print(
            f"Identified: {color_name:<6} {shape:<9} | "
            f"Pos: [{x_c:+.3f}, {y_c:+.3f}, {z_c:.3f}]m | Yaw: {yaw_angle:+.1f}°"
        )

    def process_frame(self, color_image, depth_frame):
        hsv_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
        kernel = np.ones((5, 5), np.uint8)
        combined_mask = np.zeros(hsv_image.shape[:2], dtype=np.uint8)
        detections = []

        for color_name, (lower_color, upper_color) in self.color_ranges.items():
            mask = cv2.inRange(hsv_image, lower_color, upper_color)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            combined_mask = cv2.bitwise_or(combined_mask, mask)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                if cv2.contourArea(contour) < self.min_area:
                    continue

                shape = self.classify_shape(contour)
                pose = self.extract_pose(depth_frame, contour)
                if pose is None:
                    continue

                detections.append({
                    "color": color_name,
                    "shape": shape,
                    "pose": pose,
                    "area": cv2.contourArea(contour),
                })

        return combined_mask, detections

    def get_aligned_frames(self):
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        if not depth_frame or not color_frame:
            return None, None
        color_image = np.asanyarray(color_frame.get_data())
        return color_image, depth_frame

    def get_object_camera_xyz(self, target_color=None):
        """Captures a single frame and returns [X, Y, Z] for the largest matching detection."""
        color_image, depth_frame = self.get_aligned_frames()
        if color_image is None:
            return None

        _, detections = self.process_frame(color_image, depth_frame)
        if target_color is not None:
            detections = [d for d in detections if d["color"].lower() == target_color.lower()]

        if not detections:
            return None

        largest = max(detections, key=lambda d: d["area"])
        return largest["pose"]["xyz"]

    def stream_loop(self):
        while True:
            color_image, depth_frame = self.get_aligned_frames()
            if color_image is None:
                continue

            combined_mask, detections = self.process_frame(color_image, depth_frame)
            for detection in detections:
                self.draw_detection(
                    color_image,
                    detection["color"],
                    detection["shape"],
                    detection["pose"],
                )

            cv2.imshow("RealSense Workspace Stream", color_image)
            cv2.imshow("Active Color Mask", combined_mask)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    def stop(self):
        if self.pipeline is not None:
            self.pipeline.stop()
        cv2.destroyAllWindows()


def get_object_in_world_frame(camera_xyz, T_base_to_flange, T_flange_to_camera):
    """
    Transforms an object's [X,Y,Z] from the camera lens to the real-world robot base.
    """
    # 1. Convert the 3D point to a 4x1 Homogeneous Coordinate Vector
    P_camera = np.array([[camera_xyz[0]], 
                         [camera_xyz[1]], 
                         [camera_xyz[2]], 
                         [1.0]])
    
    # 2. Transform from Camera Frame -> Wrist Flange Frame
    P_flange = np.dot(T_flange_to_camera, P_camera)
    
    # 3. Transform from Wrist Flange Frame -> Robot Base Frame
    P_base = np.dot(T_base_to_flange, P_flange)
    
    # 4. Extract the clean real-world X, Y, Z coordinates (in meters)
    world_x = float(P_base[0])
    world_y = float(P_base[1])
    world_z = float(P_base[2])
    
    return [world_x, world_y, world_z]
