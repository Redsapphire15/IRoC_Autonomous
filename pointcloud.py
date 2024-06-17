import pyrealsense2 as rs
import numpy as np
import open3d as o3d

# Initialize RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

# Start streaming
pipeline.start(config)

try:
    # Wait for a few frames to allow auto-exposure to stabilize
    for _ in range(30):
        pipeline.wait_for_frames()

    # Capture a frame
    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()

    # Convert depth frame to numpy array
    depth_image = np.asanyarray(depth_frame.get_data())

    # Create point cloud from depth image
    pc = rs.pointcloud()
    points = pc.calculate(depth_frame)
    vertices = np.asarray(points.get_vertices())

    # Filter out invalid points
    valid_indices = np.all(np.logical_not(np.isnan(vertices)), axis=1)
    vertices = vertices[valid_indices]

    # Visualize point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(vertices)
    o3d.visualization.draw_geometries([pcd])

finally:
    # Stop streaming
    pipeline.stop()
