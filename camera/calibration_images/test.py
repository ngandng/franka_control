from pathlib import Path

import cv2


# 7x5 squares CharUco board (=> 6x4 inner chess corners)
SQUARES_X = 7
SQUARES_Y = 5
INNER_CORNERS = (SQUARES_X - 1, SQUARES_Y - 1)

SQUARE_LENGTH = 0.040
MARKER_LENGTH = 0.024

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
CHARUCO_DICT_CANDIDATES = [
    "DICT_4X4_50",
    "DICT_4X4_100",
    "DICT_5X5_50",
    "DICT_5X5_100",
    "DICT_6X6_50",
    "DICT_6X6_100",
    "DICT_6X6_250",
]


def _build_charuco_board(aruco_module, dictionary):
    if hasattr(aruco_module, "CharucoBoard"):
        return aruco_module.CharucoBoard((SQUARES_X, SQUARES_Y), SQUARE_LENGTH, MARKER_LENGTH, dictionary)
    if hasattr(aruco_module, "CharucoBoard_create"):
        return aruco_module.CharucoBoard_create(
            SQUARES_X, SQUARES_Y, SQUARE_LENGTH, MARKER_LENGTH, dictionary
        )
    return None


def _detect_markers(aruco_module, gray_image, dictionary):
    if hasattr(aruco_module, "detectMarkers"):
        return aruco_module.detectMarkers(gray_image, dictionary)

    if hasattr(aruco_module, "ArucoDetector"):
        parameters = None
        if hasattr(aruco_module, "DetectorParameters"):
            parameters = aruco_module.DetectorParameters()
        elif hasattr(aruco_module, "DetectorParameters_create"):
            parameters = aruco_module.DetectorParameters_create()

        detector = aruco_module.ArucoDetector(dictionary, parameters)
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
            # OpenCV 4.13 Python binding typically expects only the image input.
            charuco_corners, charuco_ids, marker_corners, marker_ids = detector.detectBoard(gray_image)
        except cv2.error:
            # Fallback for variants that accept marker inputs.
            charuco_corners, charuco_ids, marker_corners, marker_ids = detector.detectBoard(
                gray_image,
                corners,
                ids,
            )

        if charuco_ids is None or len(charuco_ids) == 0:
            # If interpolation failed, try with detected markers via legacy path when available.
            if hasattr(aruco_module, "interpolateCornersCharuco"):
                return aruco_module.interpolateCornersCharuco(
                    markerCorners=corners,
                    markerIds=ids,
                    image=gray_image,
                    board=board,
                )

        retval = 0 if charuco_ids is None else len(charuco_ids)
        return retval, charuco_corners, charuco_ids

    return 0, None, None


def detect_charuco(gray_image):
    if not hasattr(cv2, "aruco"):
        return False, None, None, None

    aruco = cv2.aruco
    for dict_name in CHARUCO_DICT_CANDIDATES:
        dict_id = getattr(aruco, dict_name, None)
        if dict_id is None:
            continue

        dictionary = aruco.getPredefinedDictionary(dict_id)
        board = _build_charuco_board(aruco, dictionary)
        if board is None:
            continue

        corners, ids, _ = _detect_markers(aruco, gray_image, dictionary)
        if ids is None or len(ids) == 0:
            continue

        _, charuco_corners, charuco_ids = _interpolate_charuco(aruco, corners, ids, gray_image, board)
        if charuco_ids is not None and len(charuco_ids) >= 4:
            return True, dict_name, charuco_corners, charuco_ids

    return False, None, None, None


def detect_chessboard_fallback(gray_image):
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    found, corners = cv2.findChessboardCorners(gray_image, INNER_CORNERS, flags)
    if not found:
        return False, None

    refined_corners = cv2.cornerSubPix(
        gray_image,
        corners,
        winSize=(11, 11),
        zeroZone=(-1, -1),
        criteria=criteria,
    )
    return True, refined_corners


def main():
    folder = Path(__file__).resolve().parent
    image_paths = sorted(
        [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
    )

    if not image_paths:
        print(f"No images found in {folder}")
        return

    print(f"Found {len(image_paths)} images in {folder}")
    print("Controls: n/Space = next image, q/Esc = quit")

    detected_count = 0

    for index, image_path in enumerate(image_paths, start=1):
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Could not read image: {image_path.name}")
            continue

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        found, dict_name, charuco_corners, charuco_ids = detect_charuco(gray)

        display = image.copy()
        if found:
            detected_count += 1
            cv2.aruco.drawDetectedCornersCharuco(display, charuco_corners, charuco_ids, (0, 255, 0))
            status_text = f"FOUND CHARUCO ({dict_name})"
            color = (0, 255, 0)
        else:
            fallback_found, fallback_corners = detect_chessboard_fallback(gray)
            if fallback_found:
                detected_count += 1
                cv2.drawChessboardCorners(display, INNER_CORNERS, fallback_corners, fallback_found)
                status_text = f"FOUND CHESSBOARD {INNER_CORNERS[0]}x{INNER_CORNERS[1]}"
                color = (0, 255, 255)
            else:
                status_text = "NOT FOUND"
                color = (0, 0, 255)

        cv2.putText(
            display,
            f"{index}/{len(image_paths)} | {image_path.name} | {status_text}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            color,
            2,
            cv2.LINE_AA,
        )

        cv2.imshow("Chessboard Detection", display)
        key = cv2.waitKey(0) & 0xFF
        if key in (ord("q"), 27):
            break

    cv2.destroyAllWindows()
    print(f"Detected chessboard in {detected_count}/{len(image_paths)} images")


if __name__ == "__main__":
    main()