#!/usr/bin/env python3

import rospy
import cv2
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import TwistStamped
from cv_bridge import CvBridge
import numpy as np
import os

class MiRoClient:
    """
    Script to detect a face and keep it centered in the camera feed.
    """

    TICK = 0.02  # Update interval for the control loop
    SLOW = 0.1  # Rotation speed (rad/s)
    FAST = 0.3  # Forward speed (m/s)
    CAM_FREQ = 1  # Frequency of camera frame updates
    DEBUG = False  # Set to True to enable debugging visualizations
    FRAME_WIDTH = 640  # Default frame width
    FRAME_HEIGHT = 480  # Default frame height
    FACE_CASCADE_PATH = "/path/to/haarcascade_frontalface_default.xml"  # Path to OpenCV Haar Cascade XML file for face detection

    def __init__(self):
        rospy.init_node("face_follower", anonymous=True)
        rospy.sleep(2.0)  # Give some time to ensure everything is initialized
        self.image_converter = CvBridge()

        topic_base_name = "/" + os.getenv("MIRO_ROBOT_NAME")
        self.sub_cam = rospy.Subscriber(
            topic_base_name + "/sensors/caml/compressed",
            CompressedImage,
            self.callback_cam,
            queue_size=1,
            tcp_nodelay=True,
        )

        self.vel_pub = rospy.Publisher(
            topic_base_name + "/control/cmd_vel", TwistStamped, queue_size=0
        )

        self.face_cascade = cv2.CascadeClassifier(self.FACE_CASCADE_PATH)
        self.input_camera = None
        self.new_frame = False

    def callback_cam(self, ros_image):
        """
        Callback function to handle camera image.
        """
        try:
            # Convert compressed ROS image to raw CV image
            image = self.image_converter.compressed_imgmsg_to_cv2(ros_image, "rgb8")
            self.input_camera = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            self.new_frame = True
        except Exception as e:
            rospy.logerr("Error in converting image: %s", str(e))

    def detect_face(self, frame):
        """
        Detect faces in the frame using OpenCV's CascadeClassifier.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        if len(faces) > 0:
            # Assume the first detected face is the one to track
            x, y, w, h = faces[0]
            center_x = x + w // 2
            center_y = y + h // 2
            return (center_x, center_y)
        return None

    def move_robot(self, linear_speed, angular_speed):
        """
        Send velocity commands to the robot to move it.
        """
        msg_cmd_vel = TwistStamped()
        msg_cmd_vel.twist.linear.x = linear_speed
        msg_cmd_vel.twist.angular.z = angular_speed
        self.vel_pub.publish(msg_cmd_vel)

    def follow_face(self):
        """
        Main loop to follow the detected face.
        """
        print("MiRo is following a face. Press CTRL+C to stop.")
        while not rospy.is_shutdown():
            if self.new_frame:
                self.new_frame = False
                # Detect the face in the current frame
                face_position = self.detect_face(self.input_camera)
                if face_position is not None:
                    # Get the position of the face
                    center_x, center_y = face_position
                    # Calculate the deviation of the face from the center
                    error_x = center_x - self.FRAME_WIDTH // 2
                    error_y = center_y - self.FRAME_HEIGHT // 2

                    # Adjust rotation based on error in X direction
                    if abs(error_x) > 10:  # Only rotate if the error is significant
                        rotation_speed = 0.1 if error_x > 0 else -0.1
                        self.move_robot(0, rotation_speed)
                    else:
                        # Move forward if the face is centered
                        self.move_robot(self.FAST, 0)
                else:
                    # If no face is detected, rotate slowly to scan the environment
                    self.move_robot(0, self.SLOW)
            rospy.sleep(self.TICK)

if __name__ == "__main__":
    main = MiRoClient()
    main.follow_face()

