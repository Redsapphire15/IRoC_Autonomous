import cv2
import numpy as np
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import sys

class imei():
    def __init__(self):
        rospy.Subscriber("/zed2i/zed_node/left/image_rect_color, Image, self.image_callback")

    def image_callback(self, data):
        try:
            bridge = CvBridge()
            self.cv_image = bridge.imgmsg_to_cv2(data, "bgr8")
            self.cv_image = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2GRAY)
        except Exception as e:
            print(e)

# Convert image to grayscale
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRY)

    def cube_detect_cv(self):
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(self.cv_image, (5, 5), 0)

        # Perform Canny edge detection
        edges = cv2.Canny(blurred, 50, 150)

        # Find contours in the edge-detected image
        contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Iterate through detected contours
        for contour in contours:
            # Approximate the contour to a simpler polygon
            approx = cv2.approxPolyDP(contour, 0.03 * cv2.arcLength(contour, True), True)
            
            # If the contour has 4 vertices (quadrilateral)
            if len(approx) == 4:
                # Compute the bounding box for the contour
                (x, y, w, h) = cv2.boundingRect(approx)
                
                # Check if the bounding box is approximately square or rectangular
                aspect_ratio = float(w) / h
                if 0.9 <= aspect_ratio <= 1.1:
                    # Draw the bounding box around the contour
                    cv2.rectangle(self.cv_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(self.cv_image, 'Cube', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display the output image
        cv2.imshow('Cube Detection', self.cv_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def spin(self):
        while not rospy.is_shutdown():
            self.main()
            rate.sleep()


if __name__=="__main__":
    try:
        rospy.init_node("ahhhhh")
        rate = rospy.Rate(10)
        run = imei()
        run.spin()
    except KeyboardInterrupt:
        sys.exit()
