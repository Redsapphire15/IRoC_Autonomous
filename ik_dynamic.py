import rospy
import numpy as np
from std_msgs.msg import Float32MultiArray, MultiArrayLayout, MultiArrayDimension
from std_msgs.msg import Int32MultiArray, Bool
import sys
import math

# theta1=shouolder pan-base
# theta2 :shoulder lift
# theta3 :elbow-position


class IK_IROC():
    def __init__(self):
        #        self.enc_data = [0,0,0]
        #        self.d=[0,0,0,0] #joint lengths
        self.a = [0, 0.40, 0.35, 0]  # link lengths
        # substitute values of position of object which we will get from detection part
        self.px, self.py, self.pz = 0.75, 0, 0
        # rospy.init_node("arm_auto", anonymous=True)
        self.rate = rospy.Rate(10)
        self.motor_data = rospy.Publisher(
            '/auto_arm_signals', Int32MultiArray, queue_size=10)
        self.enc_data_sub = rospy.Subscriber(
            '/enc_drive', Float32MultiArray, self.enc_callback)  # the topic
        
        self.arm_goal_sub = rospy.Subscriber(
            '/arm_goal', Float32MultiArray, self.arm_goal_sub_clbk)
        self.arm_goal_given_sub = rospy.Subscriber(
            '/arm_goal_bool', Bool, self.arm_goal_given_clbk)
        self.ik_start_sub = rospy.Subscriber(
            "/ik_start", Bool, self.ik_start_callback)
        self.ik_start_drop=rospy.Subscriber("/ik_start_drop",Bool,self.ik_start_drop_calback)

        self.pranav_bool_pub = rospy.Publisher(
            '/ik_over_ah', Bool, queue_size=10)
        print("hello")
        self.enc_data = [0, 0, 0]
        self.arm_goal_coming_from_tanish = True
        self.goal = [0, 0, 0]
        self.pwm_limit = [100, 100, 75]
        self.q1, self.q2, self.q3 = 0, 0, 0
        self.om_bool = False
        self.x, self.y, self.z = 0, 0.35, 0.4
        self.tanish_count = 0
        self.listen_to_tanish = True
        self.goal_reached = False
        self.ik_start = False  # as ik shouldn't start until True ,pickup
        self.ik_start_drop=False # drop
        
            


    def ik_start_callback(self, data):
        self.ik_start = data
        print("self.ik_start", self.ik_start)

    def ik_start_drop_callback(self,data):
        self.ik_start_drop=data
        print("self.ik_start_drop",self.ik_start_drop)

    def arm_goal_given_clbk(self, msg):
        self.arm_goal_coming_from_tanish = msg.data

    def arm_goal_sub_clbk(self, msg):
        if not math.isnan(msg.data[0]) and not math.isnan(msg.data[1]) and not math.isnan(msg.data[2]):
            self.goal = msg.data
            print(f"Coming from Tanish {msg.data}")

    def angle_value(self):
        a1 = 0.4
        a2 = 0.35

        if (self.arm_goal_coming_from_tanish == False) or (self.listen_to_tanish == False):
            self.x = 0.0  # specify goal manually
            self.y = 0.35
            self.z = 0.4
        else:
            if (self.tanish_count <= 10):
                self.x = self.goal[0]
                self.y = self.goal[1]
                self.z = self.goal[2]
                # self.tanish_count = self.tanish_count + 1

        print(f"Tanish Count = {self.tanish_count}")

        x = self.x
        y = self.y
        z = self.z
        print(
            f"Going to coordinates :{x},{y},{z}, {abs((x**2+y**2+z**2-a1**2-a2**2)/(2*a1*a2))}")
        if (abs((x**2+y**2+z**2-a1**2-a2**2)/(2*a1*a2)) < 1):
            print("Angles being set for coordinates")
            self.q3 = math.pi/2
            self.q2 = -math.acos((x**2+y**2+z**2-a1**2-a2**2)/(2*a1*a2))
            self.q1 = math.atan(z/math.sqrt(x**2+y**2)) + \
                math.atan((-a2*math.sin(self.q2))/(a1+a2*math.cos(self.q2)))
        if (y != 0):
            self.q3 = math.atan(x/y)

        theta1 = self.q3
        theta2 = math.pi/2 - self.q1
        theta3 = math.pi/2 + self.q2

       # print(np.arccos(self.pz/(2*k)))
        base = (180*theta1)/math.pi
        shoulder = (180*theta2)/math.pi
        elbow = (180*theta3)/math.pi
        print(f"Goal: Base = {base}, Shoulder = {shoulder}, Elbow = {elbow}")

        print(
            f"Current: Base = {self.enc_data[0]}, Shoulder = {self.enc_data[1]}, Elbow = {self.enc_data[2]}")
        msg = Int32MultiArray()
        msg.data = [0, 0, 0, 0, 0, 0]

        msg.layout = MultiArrayLayout()
        msg.layout.data_offset = 0
        msg.layout.dim = [MultiArrayDimension()]
        msg.layout.dim[0].size = msg.layout.dim[0].stride = len(msg.data)
        msg.layout.dim[0].label = 'write'

        msg.data = [0, 0, 0, 0, 0, 0]

        if self.enc_data[2] < elbow-2:
            msg.data[0] = -self.pwm_limit[2]

        elif self.enc_data[2] > elbow+2:
            msg.data[0] = self.pwm_limit[2]

        if self.enc_data[0] < base-2:
            msg.data[1] = self.pwm_limit[0]

        elif self.enc_data[0] > base+2:
            msg.data[1] = -self.pwm_limit[0]

        if self.enc_data[1] < shoulder-2:
            msg.data[4] = -self.pwm_limit[1]

        elif self.enc_data[1] > shoulder+2:
            msg.data[4] = self.pwm_limit[1]

        # if(self.listen_to_tanish == False) and (self.arm_goal_coming_from_tanish == True):
        if self.goal_reached == True and (self.arm_goal_coming_from_tanish == True) and self.listen_to_tanish == False:
            #msg.data[3] =255 # for full grip while going up
            if self.ik_start_drop:
                msg.data[3]=-150 #gripper will open slowly
            else:
                msg.data[3]=255
            print("self.goal_reached:", self.goal_reached)
        print(
            f"Base PWM:{msg.data[1]}, Shoulder PWM: {msg.data[4]}, Elbow PWM: {msg.data[0]}")

        if abs(self.enc_data[0] - base) <= 3 and abs(self.enc_data[1] - shoulder) <= 3 and abs(self.enc_data[2] - elbow) <= 3:
            self.pranav_bool_pub.publish(True)
            print("Goal Reached")
            self.goal_reached = True
            if (self.arm_goal_coming_from_tanish == True) and (self.listen_to_tanish == True) and self.goal_reached == True:
                self.listen_to_tanish = False
                msg.data = [0, 0, 0, 0, 0, 0]
                for i in range(50):
                    # determine sign as per closing and opening
                    msg.data[3] = 255
                    self.motor_data.publish(msg)
                    print("Gripper Closing")
                    self.rate.sleep()

        else:
            self.pranav_bool_pub.publish(False)
            print("Goal Not Reached")
        if self.ik_start or self.ik_start_drop:  # this gives signals to arm motors only after receiving  ik_start==True
            print("ik_started")
            self.motor_data.publish(msg)
            print("motor_data tanish",msg.data)
            #self.rate.sleep()
        else:
            msg.data = [0, 0, 0, 0, 0, 0]
            self.motor_data.publish(msg)

    def enc_callback(self, msg):
        self.enc_data = [msg.data[3], -msg.data[2], -msg.data[5]]

    # def spin(self):
    #     while not rospy.is_shutdown():
    #             rate.sleep()
    def spin(self):
        while not rospy.is_shutdown():
            self.angle_value()
            rate.sleep()


if __name__ == "__main__":
    try:
        rospy.init_node("IK_Test")
        rate = rospy.Rate(10)
        run = IK_IROC()
        run.spin()
    except KeyboardInterrupt:
        # quit
        sys.exit()