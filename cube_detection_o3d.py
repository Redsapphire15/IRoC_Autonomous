#!/usr/bin/env python

import rospy
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
import open3d as o3d
import numpy as np
import pyzed.sl as sl

class PointCloudProcessor:
    def __init__(self):
        self.point_cloud_data = None

        # Initialize ZED camera
        init_params = sl.InitParameters()
        init_params.camera_resolution = sl.RESOLUTION.HD720
        init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE
        init_params.coordinate_units = sl.UNIT.METER
        self.zed = sl.Camera()
        if not self.zed.is_opened():
            print("Opening ZED Camera...")
            self.zed.open(init_params)

        self.subscriber = rospy.Subscriber("/zed/point_cloud/cloud_registered", PointCloud2, self.point_cloud_callback)

    def point_cloud_callback(self, msg):
        # Convert ROS point cloud message to numpy array
        gen = pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
        self.point_cloud_data = np.array(list(gen))

    def process_point_cloud(self):
        if self.point_cloud_data is not None:
            # Convert numpy array to Open3D point cloud
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(self.point_cloud_data)

            # Save point cloud data to a PCD file
            file_path = "point_cloud_data.pcd"
            o3d.io.write_point_cloud(file_path, pcd)
            rospy.loginfo(f"Point cloud data saved to: {file_path}")

            # Perform cube detection and compute side length
            # Replace the following lines with your cube detection and side length computation code
            cube_side_length = self.detect_cube_and_compute_side_length(pcd)
            rospy.loginfo(f"Detected cube side length: {cube_side_length}")

    def detect_cube_and_compute_side_length(self, pcd):
         # Perform cube detection and compute side length here
        pcd = o3d.io.read_point_cloud("point_cloud.ply")
        # Step 2: Downsample (optional)
        # For example, apply a voxel downsampling filter
        voxel_size = 0.05
        pcd_downsampled = pcd.voxel_down_sample(voxel_size)

        # Step 3: Segmentation
        # For example, use region growing segmentation
        with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
            labels = pcd_downsampled.cluster_dbscan(eps=0.02, min_points=10, print_progress=True)

        # Step 4: Extract cube
        # For example, filter objects that resemble a cube
        cube_objects = []
        for i in range(max(labels) + 1):
            cluster = pcd_downsampled.select_by_index(np.where(labels == i)[0])
            # Add conditions to check if the cluster resembles a cube
            if cluster is cube:
                cube_objects.append(cluster)

        # Step 5: Visualization
        # Visualize original point cloud
        o3d.visualization.draw_geometries([pcd])

        # Visualize detected cube
        for cube in cube_objects:
            o3d.visualization.draw_geometries([cube])

        # This is just a placeholder
        return 1.0  # Replace with actual computation


if __name__ == "__main__":
    rospy.init_node("point_cloud_processor")
    point_cloud_processor = PointCloudProcessor()

    # Process point cloud data in a loop
    rate = rospy.Rate(1)  # Adjust the rate according to your requirements
    while not rospy.is_shutdown():
        point_cloud_processor.process_point_cloud()
        rate.sleep()

    rospy.spin()
