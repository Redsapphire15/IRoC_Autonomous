import numpy as np 
import rospy
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
import pcl
import os
import subprocess

class CubeDetector():
    def __init__(self):
        self.point_cloud_data = None
        self.subscriber = rospy.Subscriber("/galileo/zed2/point_cloud/cloud_registered", PointCloud2, self.point_cloud_callback)
        
    def point_cloud_callback(self, msg):
        self.point_cloud_data = np.array(list(pc2.read_points(msg, skip_nans=True)))
        self.save_point_cloud_to_pcd()

    def save_point_cloud_to_pcd(self):
        if self.point_cloud_data is not None and len(self.point_cloud_data) > 0:
            # Convert numpy array to PCD file using pcl_ros command
            command = ["rosrun", "pcl_ros", "pointcloud_to_pcd", "input:=/galileo/zed2/point_cloud/cloud_registered", "__prefix:=/tmp/pcd/vel_"]
            subprocess.run(command)

    def perform_euclidean_clustering(self, cloud):
        # Load point cloud from file
        pcl_cloud = pcl.load(cloud)
        print(pcl_cloud)
        # Create EuclideanClusterExtraction object
        ec = pcl_cloud.make_EuclideanClusterExtraction()
        ec.set_ClusterTolerance(0.02)  # Set the tolerance for cluster extraction
        ec.set_MinClusterSize(10)       # Set the minimum number of points required for a cluster
        ec.set_MaxClusterSize(25000)    # Set the maximum number of points a cluster can have
    
        print("check1")
        # Perform clustering
        cluster_indices = pcl.vectors.Int()
        ec.Extract(cluster_indices)

        # Print number of clusters found
        print("Number of clusters:", len(cluster_indices))

        # Visualize clusters
        viewer = pcl.visualization.CloudViewer("Cluster viewer")
        viewer.showCloud(pcl_cloud, "original_cloud")
        for j, indices in enumerate(cluster_indices):
            cloud_cluster = pcl_cloud.extract(indices)
            viewer.showCloud(cloud_cluster, f"cluster{j}")
            
    def detect_cube_and_compute_side_length(self):
        folder_path = "/home/kavin/galileo2023/autonomous_codes/iroc"
            print("File exists at the specified location.")
            self.perform_euclidean_clustering("/tmp/pcd/vel_/my_cloud.pcd")
        else:
            print("File does not exist at the specified location.")
            
if __name__ == "__main__":
    rospy.init_node("cube_detector")
    cube_detector = CubeDetector()

    rate = rospy.Rate(1)  # Adjust the rate according to your requirements
    while not rospy.is_shutdown():
        cube_detector.detect_cube_and_compute_side_length()
        rate.sleep()

    rospy.spin()
