#!/usr/bin/env python3

import rospy
import cv2
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import TwistStamped
from cv_bridge import CvBridge
import numpy as np
import os
import signal
import sys

class MiRoClient:
    """
    Script to detect a face and keep it centered in the camera feed.
    """

    TICK = 0.02  # Update interval for the control loop
    MAX_ROT_SPEED = 0.3  # Max rotation speed (rad/s)
    MAX_FORWARD_SPEED = 0.3  # Max forward speed (m/s)
    KP = 0.002  # Proportional gain for horizontal error
    DEBUG = True  # Set to True to enable debugging visualizations

    def __init__(self):
        rospy.init_node("face_follower", anonymous=True)
        rospy.sleep(2.0)
        self.image_converter = CvBridge()

        topic_base_name = "/" + os.getenv("MIRO_ROBOT_NAME", "miro")
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

        # Use OpenCV's built-in Haar cascade path
        face_cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.face_cascade = cv2.CascadeClassifier(face_cascade_path)

        self.input_camera = None
        self.new_frame = False
        self.frame_width = 640
        self.frame_height = 480

        signal.signal(signal.SIGINT, self.shutdown_handler)

    def callback_cam(self, ros_image):
        try:
            image = self.image_converter.compressed_imgmsg_to_cv2(ros_image, "rgb8")
            self.input_camera = image
            self.frame_height, self.frame_width = image.shape[:2]
            self.new_frame = True
        except Exception as e:
            rospy.logerr("Error converting image: %s", str(e))

    def detect_face(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        if len(faces) > 0:
            # Sort by area (largest first)
            faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
            x, y, w, h = faces[0]
            return (x + w // 2, y + h // 2), (x, y, w, h)
        return None, None

    def move_robot(self, linear_speed, angular_speed):
        msg_cmd_vel = TwistStamped()
        msg_cmd_vel.twist.linear.x = linear_speed
        msg_cmd_vel.twist.angular.z = angular_speed
        self.vel_pub.publish(msg_cmd_vel)

    def follow_face(self):
        print("MiRo is following a face. Press CTRL+C to stop.")
        while not rospy.is_shutdown():
            if self.new_frame:
                self.new_frame = False
                face_position, bbox = self.detect_face(self.input_camera)
                if face_position is not None:
                    center_x, center_y = face_position
                    error_x = center_x - self.frame_width // 2

                    angular_speed = max(min(self.KP * error_x, self.MAX_ROT_SPEED), -self.MAX_ROT_SPEED)

                    if abs(error_x) > 10:
                        self.move_robot(0.0, angular_speed)
                    else:
                        self.move_robot(self.MAX_FORWARD_SPEED, 0.0)

                    if self.DEBUG:
                        debug_frame = self.input_camera.copy()
                        x, y, w, h = bbox
                        cv2.rectangle(debug_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        cv2.circle(debug_frame, face_position, 5, (255, 0, 0), -1)
                        cv2.imshow("Face Tracking", cv2.cvtColor(debug_frame, cv2.COLOR_RGB2BGR))
                        cv2.waitKey(1)
                else:
                    # Rotate slowly if no face is found
                    self.move_robot(0.0, 0.1)
            rospy.sleep(self.TICK)

    def shutdown_handler(self, signum, frame):
        print("\nShutting down gracefully...")
        self.move_robot(0.0, 0.0)
        if self.DEBUG:
            cv2.destroyAllWindows()
        sys.exit(0)

if __name__ == "__main__":
    main = MiRoClient()
    main.follow_face()
