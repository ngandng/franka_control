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
        self.alignconfig = rs.config()
        self.alignconfig.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.alignconfig.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        profile = self.pipeline.start(self.alignconfig)
        color_stream = profile.get_stream(rs.stream.color)
        self.intrinsics = color_stream.as_video_stream_profile().get_intrinsics()
        self.align = rs.align(rs.stream.color)


    def _get_info(self):
        # Configure and start the pipeline        
        profile = self.pipeline.start(self.alignconfig)
        
        try:
            # Get the video stream profile for the color camera
            color_stream = profile.get_stream(rs.stream.color)
            video_profile = color_stream.as_video_stream_profile()
            
            # Fetch the intrinsics object
            intrinsics = video_profile.get_intrinsics()
            
            # Print out the components of the Intrinsic Matrix
            print("\n📷 --- RealSense D435 Color Intrinsics ---")
            print(f"Resolution:  {intrinsics.width} x {intrinsics.height}")
            print(f"Focal Length: fx = {intrinsics.fx:.4f}, fy = {intrinsics.fy:.4f}")
            print(f"Principal Pt: ppx = {intrinsics.ppx:.4f}, ppy = {intrinsics.ppy:.4f}")
            print(f"Distortion Model: {intrinsics.model}")
            print(f"Distortion Coeffs: {intrinsics.coeffs}")
            
            # Construct the formal 3x3 K matrix
            print("\nFormated K Matrix:")
            print(f"[[{intrinsics.fx:.4f},   0.0000, {intrinsics.ppx:.4f}],")
            print(f" [  0.0000, {intrinsics.fy:.4f}, {intrinsics.ppy:.4f}],")
            print(f" [  0.0000,   0.0000,   1.0000]]\n")
            
        finally:
            self.pipeline.stop()


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

    def stream_loop(self, stop_event=None):
        while stop_event is None or not stop_event.is_set():
            try:
                color_image, depth_frame = self.get_aligned_frames()
            except RuntimeError:
                break
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
                if stop_event is not None:
                    stop_event.set()
                break

    def stop(self):
        if self.pipeline is not None:
            self.pipeline.stop()
        cv2.destroyAllWindows()


    def get_object_in_world_frame(self, T_base_to_flange):
        """
        Transforms an object's [X,Y,Z] from the camera lens to the real-world robot base.
        """

        camera_xyz = self.get_object_camera_xyz()
        if camera_xyz is None:
            return None

        # 1. Convert the 3D point to a 4x1 Homogeneous Coordinate Vector
        P_camera = np.array([[camera_xyz[0]], 
                            [camera_xyz[1]], 
                            [camera_xyz[2]], 
                            [1.0]])
        
        # 2. Transform from Camera Frame -> Wrist Flange Frame
        P_flange = np.dot(self.T_flange_to_camera, P_camera)
        
        # 3. Transform from Wrist Flange Frame -> Robot Base Frame
        P_base = np.dot(T_base_to_flange, P_flange)

        P_base = P_base.flatten()  # Convert from 4x1 to 1D array for easier access

        # print(f"Debug: P_camera = {P_camera}")
        # print(f"Debug: P_flange = {P_flange}")
        # print(f"Debug: P_base = {P_base}")
        
        # 4. Extract the clean real-world X, Y, Z coordinates (in meters)
        world_x = float(P_base[0])
        world_y = float(P_base[1])
        world_z = float(P_base[2])
        
        return [world_x, world_y, world_z]


    def run_hand_eye_calibration(self, robot_poses, images, chessboard_size=(6, 4), square_size=0.034):
        """
        Solve hand-eye calibration for wrist-camera transform.

        Detection reference follows camera/calibration_images/test.py:
        - Try CharUco detection first (with OpenCV API compatibility fallbacks).
        - Fall back to classic chessboard corner detection.

        chessboard_size is (inner_corners_x, inner_corners_y).
        square_size is physical inner-corner spacing in meters.
        """

        def _build_charuco_board(aruco_module, dictionary, squares_x, squares_y, marker_length):
            if hasattr(aruco_module, "CharucoBoard"):
                return aruco_module.CharucoBoard((squares_x, squares_y), square_size, marker_length, dictionary)
            if hasattr(aruco_module, "CharucoBoard_create"):
                return aruco_module.CharucoBoard_create(
                    squares_x,
                    squares_y,
                    square_size,
                    marker_length,
                    dictionary,
                )
            return None

        def _detect_markers(aruco_module, gray_image, dictionary):
            if hasattr(aruco_module, "detectMarkers"):
                return aruco_module.detectMarkers(gray_image, dictionary)

            if hasattr(aruco_module, "ArucoDetector"):
                params = None
                if hasattr(aruco_module, "DetectorParameters"):
                    params = aruco_module.DetectorParameters()
                elif hasattr(aruco_module, "DetectorParameters_create"):
                    params = aruco_module.DetectorParameters_create()
                detector = aruco_module.ArucoDetector(dictionary, params)
                return detector.detectMarkers(gray_image)

            return None, None, None

        def _interpolate_charuco(aruco_module, corners, ids, gray_image, board):
            if hasattr(aruco_module, "interpolateCornersCharuco"):
                return aruco_module.interpolateCornersCharuco(
                    markerCorners=corners,
                    markerIds=ids,
                    image=gray_image,
                    board=board,
                )

            if hasattr(aruco_module, "CharucoDetector"):
                detector = aruco_module.CharucoDetector(board)
                try:
                    charuco_corners, charuco_ids, _, _ = detector.detectBoard(gray_image)
                except cv2.error:
                    charuco_corners, charuco_ids, _, _ = detector.detectBoard(gray_image, corners, ids)
                retval = 0 if charuco_ids is None else len(charuco_ids)
                return retval, charuco_corners, charuco_ids

            return 0, None, None

        if not isinstance(chessboard_size, (tuple, list)) or len(chessboard_size) != 2:
            raise TypeError(
                f"chessboard_size must be a 2-item tuple/list of ints, got: {type(chessboard_size).__name__}"
            )

        if len(robot_poses) != len(images):
            raise ValueError(
                f"robot_poses ({len(robot_poses)}) and images ({len(images)}) must have the same length"
            )

        cols = int(chessboard_size[0])
        rows = int(chessboard_size[1])
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # Classic chessboard object points (fallback path)
        objp = np.zeros((cols * rows, 3), np.float32)
        objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_size

        # Build camera intrinsics from active RealSense profile when available.
        if self.intrinsics is not None:
            K = np.array([
                [self.intrinsics.fx, 0.0, self.intrinsics.ppx],
                [0.0, self.intrinsics.fy, self.intrinsics.ppy],
                [0.0, 0.0, 1.0],
            ], dtype=np.float64)
        else:
            K = np.array([
                [606.4210, 0.0000, 324.2198],
                [0.0000, 606.1948, 248.3232],
                [0.0000, 0.0000, 1.0000],
            ], dtype=np.float64)

        # Trackers for hand-eye input pairs.
        R_gripper2base = []
        t_gripper2base = []
        R_target2cam = []
        t_target2cam = []

        # CharUco setup derived from inner-corner grid (cols x rows)
        squares_x = cols + 1
        squares_y = rows + 1
        marker_length = square_size * 0.7
        charuco_dict_candidates = [
            "DICT_4X4_50",
            "DICT_4X4_100",
            "DICT_5X5_50",
            "DICT_5X5_100",
            "DICT_6X6_50",
            "DICT_6X6_100",
            "DICT_6X6_250",
        ]

        print("Processing frames and extracting calibration transformations...")

        valid_pairs = 0
        for i, image in enumerate(images):
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            solved = False

            # 1) CharUco-first path (same strategy as test.py)
            if hasattr(cv2, "aruco"):
                aruco = cv2.aruco
                for dict_name in charuco_dict_candidates:
                    dict_id = getattr(aruco, dict_name, None)
                    if dict_id is None:
                        continue

                    dictionary = aruco.getPredefinedDictionary(dict_id)
                    board = _build_charuco_board(aruco, dictionary, squares_x, squares_y, marker_length)
                    if board is None:
                        continue

                    marker_corners, marker_ids, _ = _detect_markers(aruco, gray, dictionary)
                    if marker_ids is None or len(marker_ids) == 0:
                        continue

                    _, charuco_corners, charuco_ids = _interpolate_charuco(
                        aruco,
                        marker_corners,
                        marker_ids,
                        gray,
                        board,
                    )
                    if charuco_ids is None or len(charuco_ids) < 4:
                        continue

                    if hasattr(board, "getChessboardCorners"):
                        board_points = board.getChessboardCorners()
                    else:
                        continue

                    ids_flat = charuco_ids.flatten().astype(np.int32)
                    obj_points = board_points[ids_flat].reshape(-1, 1, 3).astype(np.float32)
                    img_points = charuco_corners.reshape(-1, 1, 2).astype(np.float32)

                    ok, rvec, tvec = cv2.solvePnP(obj_points, img_points, K, None)
                    if not ok:
                        continue

                    R_c_t, _ = cv2.Rodrigues(rvec)
                    T_b_g = robot_poses[i]
                    R_gripper2base.append(T_b_g[0:3, 0:3])
                    t_gripper2base.append(T_b_g[0:3, 3])
                    R_target2cam.append(R_c_t)
                    t_target2cam.append(tvec)
                    valid_pairs += 1
                    solved = True
                    break

            # 2) Fallback: classic chessboard
            if not solved:
                flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
                ret, corners = cv2.findChessboardCorners(gray, (cols, rows), flags)
                if ret:
                    corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                    ok, rvec, tvec = cv2.solvePnP(objp, corners2, K, None)
                    if ok:
                        R_c_t, _ = cv2.Rodrigues(rvec)
                        T_b_g = robot_poses[i]
                        R_gripper2base.append(T_b_g[0:3, 0:3])
                        t_gripper2base.append(T_b_g[0:3, 3])
                        R_target2cam.append(R_c_t)
                        t_target2cam.append(tvec)
                        valid_pairs += 1
                        solved = True

            if not solved:
                print(f"Chessboard/CharUco detection not found in frame {i}.")

        print(f"Valid calibration pairs: {valid_pairs}/{len(images)}")

        if valid_pairs < 3:
            raise RuntimeError(
                "Not enough valid detections for hand-eye calibration. "
                f"Need at least 3, got {valid_pairs}."
            )

        R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
            R_gripper2base,
            t_gripper2base,
            R_target2cam,
            t_target2cam,
            method=cv2.CALIB_HAND_EYE_TSAI,
        )

        T_camera_to_flange = np.eye(4)
        T_camera_to_flange[0:3, 0:3] = R_cam2gripper
        T_camera_to_flange[0:3, 3] = t_cam2gripper.flatten()
        T_flange_to_camera = np.linalg.inv(T_camera_to_flange)

        print("\nCalibration complete. Flange-to-camera matrix:")
        print(np.array2string(T_flange_to_camera, separator=', '))
        return T_flange_to_camera