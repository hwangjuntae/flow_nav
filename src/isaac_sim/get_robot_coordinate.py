#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from tf2_ros import Buffer, TransformListener
from geometry_msgs.msg import TransformStamped

class RobotPositionSubscriber(Node):
    def __init__(self):
        super().__init__('robot_position_subscriber')
        # Create a buffer and transform listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.timer = self.create_timer(0.1, self.get_robot_position)  # Run every 0.1 seconds
        self.target_frame = 'base_link'  # Replace with your robot's frame
        self.source_frame = 'odom'       # Replace with your odom frame (e.g., map or odom)

    def get_robot_position(self):
        try:
            # Lookup the transform between the source frame and the target frame
            transform: TransformStamped = self.tf_buffer.lookup_transform(
                self.source_frame, self.target_frame, rclpy.time.Time()
            )
            
            # Extract position and orientation
            position = transform.transform.translation
            orientation = transform.transform.rotation
            
            self.get_logger().info(f"Robot Position: x={position.x}, y={position.y}, z={position.z}")
            # self.get_logger().info(f"Robot Orientation: x={orientation.x}, y={orientation.y}, z={orientation.z}, w={orientation.w}")
        except Exception as e:
            self.get_logger().warn(f"Could not transform {self.source_frame} to {self.target_frame}: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = RobotPositionSubscriber()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
