#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from ultralytics import YOLO
import numpy as np
from ultralytics.utils.plotting import Annotator
import imutils

class ImageSubscriber:
    def __init__(self):
        rospy.init_node('image_subscriber', anonymous=True)
        self.bridge = CvBridge()
        # self.image_sub = rospy.Subscriber('/galileo/zed2/left/image_rect_color', Image, self.image_callback)
        self.image_sub = rospy.Subscriber()
        self.window_name = 'Camera Feed'
    '''
    def image_callback(self, data):
        try:
            self.cv_image = self.bridge.imgmsg_to_cv2(data, 'bgr8')
            cv2.imshow(self.window_name, self.cv_image)
            cv2.waitKey(1)

            # Call the aruco_model method here after self.cv_image is initialized
            self.aruco_model()
        except Exception as e:
            print(e)
    '''

    def image_callback(self,data):
        try:
            self.cv_image = self.bridge.imgmsg_to_cv2(data, 'bgr8')
            gray = cv2.cvtColor (self.cv_image, cv2.COLOR_BGR2GRAY)
            edge_detection = imutils.auto_canny(gray)
            cv2.imshow(self.window_name, edge_detection)
            cv2.waitKey(1)
        except Exception as e:
            print(e)




    def aruco_model(self):
        model = YOLO("/home/kavin/Downloads/Cube_detection_IRoC/train6/weights/best.pt")
        img = np.asanyarray(self.cv_image)
        results = model.predict(self.cv_image, conf=0.25)
        for r in results:
            annotator = Annotator(img)
            boxes = r.boxes
            for box in boxes:
                b = box.xyxy[0]  # get box coordinates in (left, top, right, bottom) format
                c = box.cls
                annotator.box_label(b, model.names[int(c)])

                # Get depth data for the bounding box
                left, top, right, bottom = map(int, b)
                arrow_center = (left+right)/2
        cv2.imshow('YOLO V8 Detection', img)       

if __name__ == '__main__':
    try:
        image_subscriber = ImageSubscriber()
        rospy.spin()
    except rospy.ROSInterruptException:
        cv2.destroyAllWindows()
