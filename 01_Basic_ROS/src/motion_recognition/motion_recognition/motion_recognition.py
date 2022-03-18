import rclpy
from rclpy.node import Node

from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import Int8

import numpy as np
import cv2

class RGBToGrayConverter(Node):
    def __init__(self):
        super().__init__("convert_node")
        self.get_logger().info("Starting work")
        self.subscriber = self.create_subscription(Image, "/image_raw", self.listener_callback, 10)
        self.publisher = self.create_publisher(Int8, "/motion_flag", 10)
        self.bridge = CvBridge()
        self.avg = None
        self.thresh = None
    
    def image_processing(self, img):
        """
        сonverts to gray, applies filters for better performance -> 
        calculates the difference of two frames -> 
        converts to binary image with pixel merge
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.blur(gray, (5, 5))
    
        if self.avg is None:
            self.avg = gray.copy().astype("float")
    
        cv2.accumulateWeighted(gray, self.avg, 0.05)
        frameDelta = cv2.absdiff(gray, cv2.convertScaleAbs(self.avg))

        self.thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]
        self.thresh = cv2.dilate(self.thresh, (5, 5), iterations=2)
        
    def _check_motion(self, img):
        self.image_processing(img)
    
        limit = 0.1       
        motion_percentage = np.count_nonzero(self.thresh) / self.thresh.size

        self.get_logger().info(str(motion_percentage))

        if motion_percentage > limit:
            return 1
        else:
            return 0

    def listener_callback(self, msg: Image):
        img_from_cam = self.bridge.imgmsg_to_cv2(msg, "bgr8")

        motion_flag = self._check_motion(img_from_cam)

        msg = Int8()
        msg.data = int(motion_flag)

        self.publisher.publish(msg)


def main(args=None):
    rclpy.init(args=args)

    print("Subscriber created")
    convert_node = RGBToGrayConverter()
    rclpy.spin(convert_node)
    print("Spin")
    convert_node.destroy_node()
    print("Destroied")
    rclpy.shutdown()


if __name__ == "__main__":
    main()
