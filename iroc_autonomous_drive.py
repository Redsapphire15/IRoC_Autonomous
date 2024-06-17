#!/usr/bin/env python3

import sys 
import rospy
import math
import time
import cv2
import numpy as np
import imutils
from geometry_msgs.msg import Twist, Point
from traversal.msg import WheelRpm
from std_msgs.msg import Bool
from std_msgs.msg import Float32
from std_msgs.msg import Float32MultiArray, MultiArrayDimension, MultiArrayLayout
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator 
import open3d as o3d
from threading import Lock, Thread
from time import sleep
from nav_msgs.msg import Odometry
import torch
import argparse
#import cv_viewer.tracking_viewer as cv_viewer
from collections import defaultdict
#from iroc_center import find_center
import tf 
#from tf.transformations import quaternion_from_euler, euler_from_quaternion
from tf import LookupException,ConnectivityException,ExtrapolationException
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, Imu


#three way points before 1st goal
#consider as 3 goals only - did
#once you cross goal get some counter - do

class habibo():
	def __init__(self):

		#self.initial_pitch = None
		#self.initial_roll = None
		self.initial_yaw = 0
		#self.start_time = time.time()
		self.initial_odom = [0,0]
		self.odom_initialized = False
		
		self.model = YOLO("/home/nvidia/kavin_stuffs/train/weights/best.pt")
		self.cv_image = []
		self.current_pitch = self.current_roll = self.current_yaw = 0.0
		self.initial_pitch = self.initial_roll = self.initial_yaw = 0.0
		#get waypoint coordinates and put it in here
		self.goals_x = [1.4,3.0,0,0]
		self.goals_y = [1.7,2.3,0,0]
		self.goals_x[2] = float(input("Enter the x1 coordinate: "))
		self.goals_y[2] = float(input("Enter the y1 coordinate: "))
		
		self.goals_x[3] = float(input("Enter the x2 coordinate: "))
		self.goals_y[3] = float(input("Enter the y2 coordinate: "))
		
		rospy.Subscriber("/zed2i/zed_node/odom", Odometry, self.odom_callback)
		# rospy.Subscriber("/position_wrt_map", Float32, self.map_callback)
		#rospy.Subscriber("/zed2i/zed_node/left/image_rect_color", Image, self.image_callback)
		rospy.Subscriber("/zed2i/zed_node/rgb/image_rect_color", Image, self.image_callback)
		rospy.Subscriber("/zed2i/zed_node/depth/depth_registered", Image, self.depth_callback)
		rospy.Subscriber("/zed2i/zed_node/imu/data", Imu, self.imu_callback)
		#self.initial_pitch = self.current_pitch*180/3.14
		#self.initial_roll = self.current_roll*180/3.14
		#self.initial_yaw = self.current_yaw*180/3.14
		
		
		self.omega = 0
		self.pixel_dict = defaultdict(list)
		self.turn_completed = False #to keep track of the initial turning
		self.obs_avd_completed = False
		self.goal_counter = 0
		self.kutty_obstacle = False
		self.ignore_everything = False
		self.p_l, self.p_r, self.p_u, self.p_d = 0,0
		self.tube_pose_pub = rospy.Publisher("tube_pose",Point,queue_size=10)
		self.arm_pub = rospy.Publisher('arm_goal', Float32MultiArray, queue_size = 10)
		self.bool_pub = rospy.Publisher('arm_goal_bool', Bool, queue_size = 10)
		#self.velocity_pub = rospy.Publisher('motion',  WheelRpm, queue_size = 10)
		self.is_identified = False
		self.depth = None

		


		#  if err != sl.ERROR_CODE.SUCCESS:
		#     print(repr(err))
		#    self.zed.close()
		 #   exit(1)
		#brightness_value = 0
		self.bridge = CvBridge()

		#self.odom_self = np.array(2)
		self.odom_self = [0.0,0.0]
		# if self.goal_counter == 0:
		# 	self.goal_x = self.goals_x[0]
		# 	self.goal_y = self.goals_y[0]
		self.desired_angle = 0.0
		self.ret_cube = False   #bool for saying if crater is there
		self.ret_crater = False
		self.tanish = False
		self.p_x, self.p_y = 0,0
		self.anuj = False
		
		#self.zed.set_camera_settings(sl.VIDEO_SETTINGS.BRIGHTNESS, brightness_value)
		time.sleep(2)


	def euler_from_quaternion(self,x, y, z, w):
		"""
		Convert a quaternion into euler angles (roll, pitch, yaw)
		roll is rotation around x in radians (counterclockwise)
		pitch is rotation around y in radians (counterclockwise)
		yaw is rotation around z in radians (counterclockwise)
		"""
		t0 = +2.0 * (w * x + y * z)
		t1 = +1.0 - 2.0 * (x * x + y * y)
		roll_x = math.atan2(t0, t1)
	     
		t2 = +2.0 * (w * y - z * x)
		t2 = +1.0 if t2 > +1.0 else t2
		t2 = -1.0 if t2 < -1.0 else t2
		pitch_y = math.asin(t2)
	     
		t3 = +2.0 * (w * z + x * y)
		t4 = +1.0 - 2.0 * (y * y + z * z)
		yaw_z = math.atan2(t3, t4)
	     
		return roll_x, pitch_y, yaw_z # in radians
		
	
	def find_height(self, top, bottom, left, right, depth):
		#run continuously until find next 
		f_length = 2.1
		print("im inside find_height")
		#sensor_height = 500  # Height of the camera sensor in pixels
		object_height = top - bottom  # Object height in pixels
		print("object_height", object_height)
		pixel_height = abs(object_height)
		
		# self.zed camera provides depth in millimeters
		depth_value = self.depth_image[int((top + bottom) / 2), int((left+right)/2)]  # Depth at the center of the bounding box
		print("depth_value inside find_height", depth_value)
		#print(depth_value)
		#depth_value = depth_value[1]
		#print(depth_value)
		# # Convert depth value from millimeters to meters
		#depth_meters = depth_value / 1000.0

		# Calculate the height of the object using similar triangles
		height = pixel_height*depth_value/f_length
		
		return height

	# yolo parts 



	def cube_detect_YOLO(self):
		#here there should be an edit about how to change or wait acc to requirement. Change n here
		print("Detection")

		self.ret_obstacle = False

		# self.depth_obstacle = []
		self.depth_obstacle = 0.0

		if self.cv_image is not None:
			print("image is not none")
			img = self.cv_image
			print("Dimensions of self.cv_image", np.shape(self.cv_image))
			results = self.model.predict(img, conf=0.25, max_det=10)

			for r in results:
				annotator = Annotator(np.ascontiguousarray(img))  # Ensure image is contiguous
				boxes = r.boxes
				for box in boxes:
					b = box.xyxy[0]  # get box coordinates in (left, top, right, bottom) format
					c = box.cls
					annotator.box_label(b, self.model.names[int(c)])
					# Draw rectangle around the object
					left, top, right, bottom = map(int, b)
					cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)
					try:
						if (int(c) ==  0 or int(c) == 1):
							if (left+right)/2 < 675 or (left+ right)/2 > 45:
								depth_value = self.depth_image[ int((top+bottom)/2),int((left+right)/2)]
								print("depth_value: ", depth_value)
								# self.depth_obstacle.append(depth_value)
								self.depth_obstacle = depth_value
								height = self.find_height(top, bottom,left,right,depth_value)
								
								print('top, bottom,left,right,depth_value', top, bottom,left,right,depth_value)
								print("Height of the object:", height)
								if height <30:
									if self.goal_counter <= 1:
										x,y,z = self.posn_transformation(left, right)
										if abs(x - self.goals_x[self.goal_counter]) < 0.1 and abs(y - self.goals_y[self.goal_counter]) < 0.1:
											self.ignore_everything = True
											self.p_l, self.p_r, self.p_u, self.p_d = left, right, top, bottom 

								if abs(height)>30 and depth_value<1.5:
									self.ret_obstacle = True
									self.obstacle_avoidance(left, right)
									rospy.sleep(0.25)
						if int(c) == 2 and self.goal_counter == 1:
							# self.tanish
							# self.tanish = True
							if (left+right)/2 < 640 and (left+right)/2 > 60:
								depth_value = self.depth_image[int((top+bottom)/2), int((left+right)/2)]
								print("depth_value of cylinder", depth_value)
								if depth_value <= 1:
									self.tanish == True
						if int(c) == 3 and self.goal_counter == 2:
							self.anuj = True
							#not important rn. tc of this later
							
                        

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
		print("sum and len of depth_obstacle", sum(self.depth_obstacle), len(self.depth_obstacle))
		cord_z=sum(self.depth_obstacle)/len(self.depth_obstacle)
		x1, y1, z1 = self.frame_transformation(cord_x, cord_y, cord_z)
		return x1, y1, z1



	def obstacle_avoidance(self, x, y):
		x, y, z = self.posn_transformation(x,y)
		#obs_vec = np.array(2)
		obs_vec = [x,y]
		kp = 1.0                # proportionality constant. set to 1 for now
		print("Im inside obstacle_avoidance now")
		#negate_obs = np.array(2)
		negate_obs = [-x, -y]
		print("negate_obs", negate_obs)
		#goal_vec = np.array(2)
		goal_vec = [self.goal_x, self.goal_y]
		unit_nobs = []
		#unit_nobs.append(negate_obs[0]/(x**2 + y**2))
		unit_nobs.append(negate_obs[1]/(0.1*(x**2 + y**2)))
		unit_nobs.append(negate_obs[0]/(0.1*(x**2 + y**2)))
		print("unit_nobs", unit_nobs)
		#heading_vec = np.array(2)
		heading_vec = [goal_vec[0] + unit_nobs[0], goal_vec[1] + unit_nobs[1]]
		print("heading_vec", heading_vec)
		desired_turn = math.atan(heading_vec[1]/heading_vec[0])*360/3.14
		print(f"Desired turn in obstcl avd = {desired_turn}")
		g = WheelRpm()
		self.omega = int(kp*(self.current_yaw - desired_turn))
		g.vel= 10          
		print("desired_turn - self.current_yaw in obstacle avoidance", desired_turn - self.current_yaw)
		if abs(desired_turn - self.current_yaw) > 5:
			print("self.omega",self.omega)
			
			if self.omega>0:
				g.omega = min(self.omega, 25)
			else:
				g.omega = max(self.omega, -25)
		
		else:
			print("self.omega", self.omega)
			g.omega = 0
		self.obs_avd_completed = True
		vel_pub.publish(g)
		self.turn_completed = False

#		rospy.sleep(0.25)
		return heading_vec
	def safe_obstacle_avoidance(self,x,y):
		if self.ignore_everything:
			#bro has to make sure it turns and goes in between them
			#frame-wise approach should be there until last minute calculate ah
			g = WheelRpm()
			g.vel = 0
			g.omega = 0
			vel_pub.publish(g)
			rospy.sleep(1)
	def show_coordinates(self,p_x,p_y,cv_image):
        #K matrix values
		f_x=527.2972398956961
		f_y=527.2972398956961
		c_x=300.0
		c_y=180.0
		self.cord_x=self.depth*(p_x-c_x)/f_x
		self.cord_y=self.depth*(p_y-c_y)/f_y
		self.cord_z=self.depth
		self.font=cv2.FONT_HERSHEY_SIMPLEX
		self.font_scale=0.5
		self.color=[255,0,0]
		self.thickness=1
	
			
        

	def frame_transformation(self, x, y, z):
		br = tf.TransformBroadcaster()
		br.sendTransform((x,y,z), (0,0,0,1), rospy.Time.now(), "obstacle", "zed2i_base_link")
		listener = tf.TransformListener()
		trans = []
		rospy.sleep(0.1)
		try:
			(trans, rot) = listener.lookupTransform("map", "obstacle", rospy.Time(0))
			print("trans", trans)
			# br.sendTransform(trans, (0,0,0,1), rospy.Time.now(),)
		except (LookupException, ConnectivityException, ExtrapolationException) as e:
			print(e)
			pass
		return trans[0], trans[1], trans[2]


	def go_to_goal(self,n):
		#waypoint navigation within this
		kp = 1.0
		kp_linear = 25
		#goal_vec = np.array(2)
		#print("goal_vec:",goal_vec)
		goal_vec = [self.goals_x[n], self.goals_y[n]]
		print("goal_vec: ", goal_vec)
		desired_turn = math.atan((goal_vec[1]-self.odom_self[1])/(goal_vec[0]-self.odom_self[0]))
		desired_turn = (desired_turn)*180/3.14
		print("desired turn in gtg", desired_turn)
		goal_distance = math.sqrt(((goal_vec[0] - self.odom_self[0])**2) + (goal_vec[1] - self.odom_self[1])**2)
		print("goal distance in metres", goal_distance)
		self.omega = int(kp*(self.current_yaw - desired_turn))
		desired_vel = int(kp_linear * goal_distance)
		g = WheelRpm()
		g.vel = min(desired_vel, 25)
		if goal_distance < 0.5:
			g.vel = 0
			self.goal_counter += 1
		# if self.goal_counter == 1:
        #                 print("First Goal Reached :)")
        #                 self.goal_x = self.goals_x[1]
        #                 self.goal_y = self.goals_y[1]	
		if self.obs_avd_completed == True:
			print("self.obs_avd_completed", self.obs_avd_completed)
			g.omega = 0
			g.vel = 25
			vel_pub.publish(g)
			rospy.sleep(2)
			self.obs_avd_completed = False
		else:
			print("self.current_yaw - desired_turn", self.current_yaw - desired_turn)
			if abs(self.current_yaw - desired_turn) > 3:
				print("self.omega gtg", self.omega)
				if self.omega>0:
					g.omega = min(self.omega, 25)
				else:
					g.omega = max(self.omega, -25)
			elif abs(self.current_yaw - desired_turn) < 3:
				print("self.omega gtg", 0)
				g.omega = 0
		vel_pub.publish(g)

	def main(self):
		if self.ignore_everything == True:
			self.go_to_goal(n)
		self.cube_detect_YOLO()
		n = 0
		#cube_thread.start()
		print("check1 main")
		print("self.odom = ", self.odom_self)
		print("self.orientation", self.current_yaw)
		print("initial_orientation", self.initial_yaw)
		#rospy.sleep(1)
		#self.crater_detect_YOLO()
		#rospy.sleep(1)
		#        crater_thread.start()
		
		print("self.ret_cube: ", self.ret_cube)
		n = self.goal_counter
		if self.ret_crater == False and self.ret_cube == False:
			self.go_to_goal(n)   
			print("end of main")
	def main2(self):
		if self.is_identified == False:
			g = WheelRpm()
			g.vel = 20
			vel_pub.publish(g)
		if(self.image_arrived == True):
			# self.p_x,self.p_y=(self.c
			print(f"self.p_x = {self.p_x}, self.p_y = {self.p_y}")
			if self.p_x is not None and self.p_y is not None:
				self.depth = self.depth_image[int(self.p_y), int(self.p_x) ]
				self.show_coordinates(self.p_x,self.p_y,self.cv_image)
				self.tube_frame(self.cord_x,self.cord_y,self.cord_z)
				cv2.putText(self.cv_image,str((self.cord_x,self.cord_y,self.cord_z)),(int(self.p_x),int(self.p_y)),self.font,self.font_scale,self.color,self.thickness,cv2.LINE_AA)
				goal_coord = [self.cord_x, self.cord_y, self.cord_z]		
				goal_coord = np.asanyarray(goal_coord)
				msg= Float32MultiArray()
				msg.data=[0,0,0]
				msg.layout = MultiArrayLayout()
				msg.layout.data_offset = 0
				msg.layout.dim = [MultiArrayDimension()]
				msg.layout.dim[0].size = msg.layout.dim[0].stride = len(msg.data)
				msg.layout.dim[0].label = 'write'
				msg.data = [0,0,0]            
				msg.data[1] = self.cord_z #offset for zed camera
				msg.data[2] = -self.cord_y-0.35
				msg.data[0] = self.cord_x
				self.is_identified = True
				d = math.pow(self.cord_x**2 + self.cord_y**2 + self.cord_z**2, 0.5)
				g = WheelRpm()
				kp = 26.67
				if d > 0.75:
					g.vel = 20
				elif d > 0.4 and d < 0.75:
					g.vel = int(min(kp * d, 20))
				else:
					g.vel = 0
				if self.p_x > 400:
					g.omega = 20
				elif self.p_x <320 and self.p_x>0:
					g.omega = -20
				else:
					g.omega = 0
				print("G is amazing", g)
				vel_pub.publish(g)
				if not math.isnan(msg.data[0]) and not math.isnan(msg.data[1]) and not math.isnan(msg.data[2]):
					self.arm_pub.publish(msg)
					self.bool_pub.publish(True)
				else:
					pass
			cv2.imshow('frame',self.cv_image)
			if cv2.waitKey(1) & 0xFF == ord('q'):
				cv2.destroyAllWindows()
	def tube_frame(self,x,y,z):
		# self.tube_pose_pub.publish(t)
		# print(x, y, z)
		br = tf.TransformBroadcaster()
		br.sendTransform((x,y,z), (0,0,0,1), rospy.Time.now(), "sample_tube", "zed2_left_camera_optical_frame")
		listener = tf.TransformListener()
		rospy.sleep(2)
		print(2)
		try:
			(trans,rot)=listener.lookupTransform("base_link","sample_tube",rospy.Time(0))
			print(trans)
			br.sendTransform(trans,(0,0,0,1),rospy.Time.now(),"sample_tube_base","base_link")
		except (LookupException, ConnectivityException, ExtrapolationException) as e:
			print(1)
			pass



	# callbacks
	def odom_callback(self, data):
		self.odom_self[0] = data.pose.pose.position.x
		self.odom_self[1] = -data.pose.pose.position.y
		if self.odom_initialized == False:
			self.initial_odom[0] = self.odom_self[0]
			self.initial_odom[1] = self.odom_self[1]
			self.odom_initialized = True
		self.odom_self[0] = self.odom_self[0] - self.initial_odom[0]
		self.odom_self[1] = self.odom_self[1] - self.initial_odom[1]		
		# self.odom_self[2] = data.z

	def imu_callback(self, data):
		current_x = data.orientation.x
		current_y = data.orientation.y
		current_z = data.orientation.z
		current_w = data.orientation.w
		self.current_pitch, self.current_roll, self.current_yaw = self.euler_from_quaternion(current_x, current_y, current_z, current_w)
		
		if self.initial_yaw == 0:
			#self.current_pitch = self.current_pitch*180/3.14 - self.initial_pitch
			#self.current_roll = self.current_roll*180/3.14 - self.initial_roll
			self.initial_yaw = self.current_yaw*180/3.14
		self.current_yaw = -(self.current_yaw * 180/3.14) + self.initial_yaw
		if self.current_yaw < -120:
			self.current_yaw = self.current_yaw + 360
		elif self.current_yaw > 120:
			self.current_yaw = self.current_yaw - 360
			

	def callback(self,data):
		try:
			self.cv_image=self.bridge.imgmsg_to_cv2(data,'bgr8')
			self.image_arrived = True
		except Exception as e :
			print(e)    

				



	def image_callback(self, data):
		try:
			self.cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
		except Exception as e:
			print(e)

	def depth_callback(self, data):
		try:
			self.depth_image = self.bridge.imgmsg_to_cv2(data, "passthrough")
		except Exception as e:
			print(e)

	def spin(self):
		while not rospy.is_shutdown():
			if self.tanish == False:
				self.main()
				rate.sleep()
			else:
				self.main2()
				rate.sleep()



if __name__ == "__main__":
	try:
		#get goal location. I have no clue how to provide goal location and stuff
		#not anymore bruh
		rospy.init_node("hihiih")
		vel_pub = rospy.Publisher("motion", WheelRpm, queue_size=10)
		rate = rospy.Rate(10)
		run = habibo()
		run.spin()
	except KeyboardInterrupt:
		sys.exit()



# ID Type X, Y, Z (meters)
# 1 A1 1.6, -0.9, 0.0
# 2 A1 1.4, -1.7, 0.0
# 3 A1 2.5, -1.4, 0.0
# 4 A1 3.0, -2.3, 0.0
# 5 A1 4.7, -1.8, 0.0
# 6 A1 7.58, -2.7, 0.0
# 7 A2 1.2, -2.5, 0.0
# 8 A2 2.0, -0.6, 0.0
# 9 A2 3.0, -0.5, 0.0
# 10 A2 4.5, -1.25, 0.0
# 11 A2 7.38, -1.0, 0.0
# 12 B1 2.3, -2.2, 0.0
# 13 B1 4.7, -2.4, 0.0
# 14 B1 7.58, -1.8, 0.0
# 15 B2 2.7, -0.7, 0.0
# 16 B2 4.5, -3.0, 0.0
# 17 B2 8.58, -0.9, 0.0
# SP Start Position 0, -1.2, 0
# T Sample Tube 3.3, -0.825, 0
# WP Way Point 2.9, -2.1, 0
# C & FP Sample Container & Final Position 9.25, -3.25, 0)