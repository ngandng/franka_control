import pyrealsense2 as rs

def main():
    # Configure and start the pipeline
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    
    profile = pipeline.start(config)
    
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
        pipeline.stop()

if __name__ == "__main__":
    main()