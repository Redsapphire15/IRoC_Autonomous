#!/usr/bin/env python3

import sys 
import rospy
import math
import time
import cv2
import numpy as np
import imutils
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool
from std_msgs.msg import Float32
from std_msgs.msg import Float32MultiArray
import pyzed.sl as sl
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator 
import open3d as o3d
from threading import Lock, Thread
from time import sleep
import torch
import argparse
import cv_viewer.tracking_viewer as cv_viewer
from collections import defaultdict



class habibo:
    def __init__(self):
        rospy.Subscriber("/visual_odom_topic", Float32, self.odom_callback)
        rospy.Subscriber("/position_wrt_map", Float32, self.map_callback)
        rospy.Subscriber("/imu", Float32MultiArray, self.imu_callback)
        # rospy.Subscriber("/map_topic", Float32. 10)
        self.zed = sl.Camera()
        input_type = sl.InputType()
        init_params = sl.InitParameters()
        init_params.camera_resolution = sl.RESOLUTION.HD720
        init_params.coordinate_units = sl.UNIT.METER
        init_params.depth_mode = sl.DEPTH_MODE.ULTRA  # QUALITY
        init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
        init_params.depth_maximum_distance = 50
        init_params.camera_fps = 30
        runtime_params = sl.RuntimeParameters()
        err = self.zed.open(init_params)
        positional_tracking_parameters = sl.PositionalTrackingParameters()
        self.zed.enable_positional_tracking(positional_tracking_parameters)
        
        self.goal_x = float(input("Enter the x coordinate: "))
        self.goal_y = float(input("Enter the y coordinate: "))
        self.current_yaw = 0.0
        self.desired_angle = math.tan(float(self.goal_y/self.goal_x))
        self.omega = 0.0
        self.pixel_dict = defaultdict(list)
        
        if err != sl.ERROR_CODE.SUCCESS:
            print(repr(err))
            self.zed.close()
            exit(1)
        brightness_value = 0


        self.odom_self = np.array(2)
        self.ret_cube = False   #bool for saying if crater is there
        self.ret_crater = False
        self.zed.set_camera_settings(sl.VIDEO_SETTINGS.BRIGHTNESS, brightness_value)
        time.sleep(2)


#my special touch broooo
    def find_height(self, top, bottom, left, right, depth):
        f_length = 2.1 
        #sensor_height = 500  # Height of the camera sensor in pixels
        object_height = top - bottom  # Object height in pixels
        pixel_height = (top-bottom)*0.26
        # self.zed camera provides depth in millimeters
        depth_value = depth.get_value(int((top + bottom) / 2), int((left+right)/2))  # Depth at the center of the bounding box
        print(depth_value)
        depth_value = depth_value[1]
        print(depth_value)
    # # Convert depth value from millimeters to meters
        #depth_meters = depth_value / 1000.0
        
        # Calculate the height of the object using similar triangles
        height = pixel_height*depth_value/f_length
        
        return height

# yolo parts 
    def crater_detect_YOLO(self):
        # Capture frames
        self.model1 = YOLO('/home/kavin/Downloads/crater_weights/best.pt')
        runtime_parameters = sl.RuntimeParameters()
        left_image = sl.Mat()
        depth_map = sl.Mat()
        self.ret_crater = False
        self.pixel_dict = defaultdict
        if self.zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
            # Retrieve left image and depth map
            self.zed.retrieve_image(left_image, sl.VIEW.LEFT)
            self.zed.retrieve_measure(depth_map, sl.MEASURE.DEPTH)

            # Convert self.zed image to numpy array
            img = np.ascontiguousarray(left_image.get_data()[:, :, :3])

            # Run YOLO model inference
            results = self.model1.predict(img, conf=0.45, max_det=3)

            for r in results:
                annotator = Annotator(np.ascontiguousarray(img))  # Ensure image is contiguous
                boxes = r.boxes
                for box in boxes:
                    b = box.xyxy[0]  # get box coordinates in (left, top, right, bottom) format
                    c = box.cls
                    annotator.box_label(b, self.model1.names[int(c)])

                    # Draw rectangle around the object
                    left, top, right, bottom = map(int, b)
                    cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)
                    #self.ret_crater = True
                    try:
                        depth_value = depth_map.get_value(int((left+right)/2), int((top+bottom)/2))
                        self.depth_crater.append(depth_value)
                        self.pixel_values.append((left, right))
                        height = self.find_height(self.depth_crater[-1], top, bottom,left,right)
                        print("Height of the object:", height)
                        if height >= 30 and depth_value <=1.5:
                            self.ret_crater = True
                            self.obstacle_avoidance(left, right)
                    except Exception as e:
                        print("Error:", e)

            # Show annotated image
            cv2.imshow('YOLO V8 Detection', img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                return
        



    def cube_detect_YOLO(self):
        self.model2 = YOLO("/home/kavin/Downloads/Cube_detection_IRoC/updated_weights/best.pt")
        runtime_parameters = sl.RuntimeParameters()
        left_image = sl.Mat()
        depth_map = sl.Mat()
        self.ret_cube = False
        if self.zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
            # Retrieve left image and depth map
            self.zed.retrieve_image(left_image, sl.VIEW.LEFT)
            self.zed.retrieve_measure(depth_map, sl.MEASURE.DEPTH)
            self.depth_crater = []
            self.pixel_values = []
            # Convert self.zed image to numpy array
            img = np.ascontiguousarray(left_image.get_data()[:, :, :3])

            # Run YOLO model inference
            results = self.model1.predict(img, conf=0.45, max_det=3)

            for r in results:
                annotator = Annotator(np.ascontiguousarray(img))  # Ensure image is contiguous
                boxes = r.boxes
                for box in boxes:
                    b = box.xyxy[0]  # get box coordinates in (left, top, right, bottom) format
                    c = box.cls
                    annotator.box_label(b, self.model1.names[int(c)])
                    # self.ret_cube = True
                    # Draw rectangle around the object
                    left, top, right, bottom = map(int, b)
                    cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)
                    try:
                        
                        depth_value = depth_map.get_value(int((left+right)/2), int((top+bottom)/2))
                        self.depth_crater.append(depth_value)
                        self.pixel_values.append((left, right))
                        height = self.find_height(self.depth_crater[-1], top, bottom,left,right)
                        print("Height of the object:", height)
                        if abs(height)>30 and depth_value<1.5:
                            self.ret_cube = True
                            self.obstacle_avoidance(left, right)

                    except Exception as e:
                        print("Error:", e)

            # Show annotated image
            cv2.imshow('YOLO V8 Detection', img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                return

    def posn_transformation(self, p_x, p_y):
        #K matrix values
        f_x=527.2972398956961
        f_y=527.2972398956961
        c_x=658.8206787109375
        c_y=372.25787353515625
        cord_x=(p_x-c_x)/f_x
        cord_y=(p_y-c_y)/f_y
        cord_z=self.depth_crater
        return cord_x, cord_y, cord_z 
    


    def obstacle_avoidance(self, x, y):
        x, y, z = self.posn_transformation(x,y)
        obs_vec = np.array(2)
        obs_vec = [x,y]
        kp = 1.0                # proportionality constant. set to 1 for now
        negate_obs = np.array(2)
        negate_obs = [-x, -y]
        goal_vec = np.array(2)
        goal_vec = [self.goal_x, self.goal_y]
        unit_nobs = negate_obs/(math.sqrt(x**2 + y**2))
        heading_vec = np.array(2)
        heading_vec = [goal_vec[0] + unit_nobs[0], goal_vec[1] + unit_nobs[1]]
        desired_turn = math.tan(heading_vec[1]/heading_vec[0])
        g = Twist()
        self.omega = kp*desired_turn
        g.linear.x = 0.0           # I'm thinking of stopping and turning.. will see how it works out to move and turn
        if desired_turn>=0:
            g.angular.z = min(self.omega, 0.05)
        else:
            g.angular.z = max(self.omega, -0.05)
        vel_pub.publish(g)
        return heading_vec
    
    def go_to_goal(self):
        kp = 1.0
        kp_linear = 1.0
        goal_vec = np.array(2)
        goal_vec = [self.goal_x, self.goal_y]
        desired_turn = math.tan((goal_vec[1]-self.odom_self[1])/(goal_vec[0]-self.odom_self[0]))
        goal_distance = math.sqrt((self.goal_x - self.odom_self[0]**2) + (self.goal_y - self.odom_self[1])**2)
        self.omega = kp*desired_turn
        desired_vel = kp_linear * goal_distance
        g = Twist()
        g.linear.x = min(desired_vel, 0.2)
        if desired_turn >= 0:
            g.angular.z = min(0.05, self.omega)
        else:
            g.angular.z = max(-0.05, self.omega)
        vel_pub.publish(g)
        
    def main(self):
        cube_thread = Thread(target=self.cube_detect_YOLO, args = (1,))
        cube_thread.start()
        crater_thread = Thread(target = self.crater_detect_YOLO, args=(1,))
        crater_thread.start()
        if self.ret_crater == False and self.ret_cube == False:
            self.go_to_goal()   

    


# callbacks
    def odom_callback(self, data):
        self.odom_self[0] = data.x
        self.odom_self[1] = data.y
        # self.odom_self[2] = data.z

    def imu_callback(self, data):
        self.current_yaw = data.z
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    opt = parser.parse_args()
    #get goal location. I have no clue how to provide goal location and stuff
    vel_pub = rospy.Publisher("cmd_vel", Twist, queue_size=10)

    # with torch.no_grad():
    #     main()