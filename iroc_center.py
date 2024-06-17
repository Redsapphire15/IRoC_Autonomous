#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point
import numpy as np
import cv2
import imutils
import tf
from cv_bridge import CvBridge
from tf import LookupException,ConnectivityException,ExtrapolationException
class find_center():
    def __init__(self):
        self.up=np.array([130,255,255])
        self.down=np.array([110,50,50])
        rospy.Subscriber('/galileo/zed2/left/image_rect_color', Image, self.get_cframe)
        rospy.Subscriber('/galileo/zed2/depth/depth_registered', Image, self.get_dframe)
        self.rate = rospy.Rate(10)
    def get_transform(self,msg):
        self.t=msg.transform.translation
        print(self.t)
    def get_dframe(self,msg):
        bridge=CvBridge()
        self.dframe=msg
        self.dframe=bridge.imgmsg_to_cv2(self.dframe, "passthrough")
    def get_cframe(self,msg):
        bridge=CvBridge()
        self.cframe=msg
        self.cframe=bridge.imgmsg_to_cv2(self.cframe, "bgr8")
        self.contour()
    def contour(self):
        blue_hsv=cv2.cvtColor(self.cframe,cv2.COLOR_BGR2HSV)
        mask=cv2.inRange(blue_hsv,self.down,self.up)
        ker=np.ones((7,7),np.uint8)
        mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,ker)
        mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,ker)
        cont=cv2.findContours(mask.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cont=imutils.grab_contours(cont)
        area=[]
        Y_array=[]
        self.D_array=[]
        for i in cont:
            area.append(cv2.contourArea(i))
        if area==[]:
            pass
        else:
            z=max(area)
            i=area.index(z)
            app=cv2.approxPolyDP(cont[i],0.005*cv2.arcLength(cont[i],True),True)
            self.x,self.y,self.w,self.h=cv2.boundingRect(app)
            i1=cv2.rectangle(self.cframe,(self.x,self.y),(self.x+self.w,self.y+self.h),(255,0,0),2,0)
            Y=self.y+1
            d1=self.get_depth(self.x+self.w/2,Y-1)
            while Y<=self.y+self.h:
                Y_array.append(Y)
                d=self.get_depth(self.x+self.w/2,Y)
                self.D_array.append(d1-d)
                d1=d
                Y=Y+1
            self.D_array=self.D_array[:len(self.D_array)-1]
            Y_array=Y_array[:len(Y_array)-1]
            self.check=[]
            for i in range(len(self.D_array)):
                if i==len(self.D_array)-1:
                    break
                else:
                    if (self.D_array[i]>=0 and self.D_array[i+1]<=0) or (self.D_array[i+1]>=0 and self.D_array[i]<=0):
                        self.check.append(i+1)
            self.p_x=int(self.x+self.w/2)
            self.p_y=int((Y_array[self.check[0]]+Y_array[self.check[1]])/2)
            cv2.circle(self.cframe,(self.p_x,self.p_y),7,(0,0,0),-1)
            self.d1=self.get_depth(self.x+self.w/2,Y_array[self.check[0]])
            self.d2=self.get_depth(self.x+self.w/2,Y_array[self.check[1]])
            self.get_coords()
            self.collectiontube_frame(self.cord_x,self.cord_y,self.cord_z)
    def get_depth(self,x,y):
        depth=self.dframe[int(y)][int(x)]
        return depth
    def get_coords(self):
        f_x=527.2972398956961
        f_y=527.2972398956961
        c_x=658.8206787109375
        c_y=372.25787353515625
        self.d=np.sqrt(0.5*(self.d1**2+self.d2**2-(0.15**2)/2))
        self.cord_x=self.d*(self.p_x-c_x)/f_x
        self.cord_y=self.d*(self.p_y-c_y)/f_y
        self.cord_z=self.d
    def collectiontube_frame(self,x,y,z):
        t=Point()
        t.x=x
        t.y=y
        t.z=z
        br=tf.TransformBroadcaster()
        br.sendTransform((x,y,z),(0.5,0.5,0.5,0.5),rospy.Time.now(),"collection_tube_frame","zed2_left_camera_optical_frame")
        listener = tf.TransformListener()
        rospy.sleep(2)
        try:
            (trans,rot)=listener.lookupTransform("base_link","collection_tube_frame",rospy.Time(0))
            print(trans)
            br.sendTransform(trans,(0.5,0.5,0.5,0.5),rospy.Time.now(),"collection_tube_base","base_link")
        except (LookupException, ConnectivityException, ExtrapolationException):
            pass
    def spin(self):
        self.main()
        self.rate.sleep()
    def main(self):
        while not rospy.is_shutdown():
            rospy.spin()
if __name__ == '__main__':
    try:
        rospy.init_node('iroc_centre', anonymous=True)
        rate = rospy.Rate(10)
        auto = find_center()
        auto.spin()
    except rospy.ROSInterruptException:
        pass
