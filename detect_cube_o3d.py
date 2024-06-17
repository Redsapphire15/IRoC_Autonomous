#!/usr/bin/env python

import rospy
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
import pcl
import numpy as np
import open3d as o3d

class CubeDetector:
    def __init__(self):
        self.point_cloud_data = None
        self.subscriber = rospy.Subscriber("/galileo/zed2/point_cloud/cloud_registered", PointCloud2, self.point_cloud_callback)

    def point_cloud_callback(self, msg):
        # Convert ROS point cloud message to numpy array
        self.point_cloud_data = np.array(list(pc2.read_points(msg, skip_nans=True)))
        pcl.save(self.point_cloud_data,"ahh.pcd")


    def perform_euclidean_clustering(self, cloud):
        # Convert point cloud to PCL point cloud
        pcl_cloud = pcl.PointCloud()
        pcl_cloud.from_array(np.float32(cloud))

        # Create EuclideanClusterExtraction object
        ec = pcl_cloud.make_EuclideanClusterExtraction()
        ec.set_ClusterTolerance(0.02)  # Set the tolerance for cluster extraction
        ec.set_MinClusterSize(10)       # Set the minimum number of points required for a cluster
        ec.set_MaxClusterSize(25000)    # Set the maximum number of points a cluster can have

        # Perform clustering
        cluster_indices = pcl.vectors.Int()
        ec.Extract(cluster_indices)

        # Visualize clusters
        viewer = pcl.visualization.CloudViewer("Cluster viewer")
        viewer.showCloud(pcl_cloud, "original_cloud")
        for j, indices in enumerate(cluster_indices):
            cloud_cluster = pcl_cloud.extract(indices)
            viewer.showCloud(cloud_cluster, f"cluster{j}")

    def detect_cube_and_compute_side_length(self):
        if self.point_cloud_data is not None and len(self.point_cloud_data) > 0:
            # Perform Euclidean clustering
            cloud = o3d.io.read_point_cloud("/home/kavin/ahh.pcd")
            self.perform_euclidean_clustering(self.point_cloud_data)
        else:
            rospy.loginfo("Trying to get point cloud data")

if __name__ == "__main__":
    rospy.init_node("cube_detector")
    cube_detector = CubeDetector()

    rate = rospy.Rate(1)  # Adjust the rate according to your requirements
    while not rospy.is_shutdown():
        cube_detector.detect_cube_and_compute_side_length()
        rate.sleep()

    rospy.spin()
