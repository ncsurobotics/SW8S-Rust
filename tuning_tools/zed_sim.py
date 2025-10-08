import rclpy
from rclpy.executors import ExternalShutdownException

from geometry_msgs.msg import PoseStamped
from builtin_interfaces.msg import Time
from datetime import datetime

def main(args=None):
    try:
        rclpy.init(args=args)
        node = rclpy.create_node('zed_sim')
        publisher = node.create_publisher(PoseStamped, '/zed/pose', 10)

        msg = PoseStamped()
        i = 0

        def timer_callback():
            nonlocal i
            i += 1
            msg.header.frame_id = "base_link"
            msg.header.stamp.sec = datetime.now().second
            msg.header.stamp.nanosec = i
            msg.pose.position.x = 1.0
            node.get_logger().info(f"Publishing {i}")
            publisher.publish(msg)

        timer_period = 1/60.0
        timer = node.create_timer(timer_period, timer_callback)
        timer

        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass

if __name__ == '__main__':
    main()
