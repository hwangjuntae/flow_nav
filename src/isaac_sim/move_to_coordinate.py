#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from tf2_ros import Buffer, TransformListener
from geometry_msgs.msg import Twist, TransformStamped
import math

class MoveToTarget(Node):
    def __init__(self):
        super().__init__('move_to_target')

        # /cmd_vel 토픽 퍼블리셔
        self.publisher = self.create_publisher(Twist, '/cmd_vel', 400)

        # TF2 버퍼와 리스너 설정
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # 목표 위치와 허용 오차 설정
        self.target_x = 55.0
        self.target_y = 0.0
        self.tolerance = 0.1  # 허용 오차를 10cm로 설정

        # 현재 위치와 방향 초기화
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_theta = 0.0
        self.reached_goal = False

        # 속도 업데이트 주기 타이머 (0.1초)
        self.timer = self.create_timer(0.1, self.update_velocity)

        # 프레임 이름
        self.source_frame = 'odom'  # 원본 프레임 (필요에 따라 map으로 변경)
        self.target_frame = 'base_link'  # 로봇의 기준 프레임

    def get_robot_position(self):
        try:
            # TF2를 통해 odom 프레임에서 base_link 프레임으로의 변환 얻기
            transform: TransformStamped = self.tf_buffer.lookup_transform(
                self.source_frame, self.target_frame, rclpy.time.Time()
            )
            
            # 위치 및 방향 추출
            position = transform.transform.translation
            orientation = transform.transform.rotation
            
            self.current_x = position.x
            self.current_y = position.y
            self.current_theta = self.get_yaw_from_quaternion(orientation)

            # 현재 위치 출력
            self.get_logger().info(
                f"Current Position: x={self.current_x:.2f}, y={self.current_y:.2f}, theta={math.degrees(self.current_theta):.2f}°"
            )
            return True
        except Exception as e:
            self.get_logger().warn(f"Could not transform {self.source_frame} to {self.target_frame}: {e}")
            return False

    def get_yaw_from_quaternion(self, q):
        # 쿼터니언을 yaw(회전각)으로 변환
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def calculate_distance(self):
        # 현재 위치와 목표 지점 간의 유클리드 거리 계산
        return math.sqrt((self.target_x - self.current_x)**2 + (self.target_y - self.current_y)**2)

    def calculate_angle_to_target(self):
        # 로봇 방향과 목표 방향 간의 각도 계산
        return math.atan2(self.target_y - self.current_y, self.target_x - self.current_x)

    def update_velocity(self):
        if self.reached_goal:
            # 목표에 도달하면 로봇 정지
            twist = Twist()
            self.publisher.publish(twist)
            return

        # 로봇 위치 업데이트 시도
        if not self.get_robot_position():
            return  # TF2로 현재 위치를 가져올 수 없으면 업데이트 중단

        # 목표 지점까지의 거리 계산
        distance = self.calculate_distance()

        if distance <= self.tolerance:
            # 목표 도달 시 속도 0으로 설정 후 멈춤
            self.get_logger().info("Goal reached. Stopping robot.")
            self.reached_goal = True
            twist = Twist()
            twist.linear.x = 0.0
            twist.angular.z = 0.0
            self.publisher.publish(twist)
            return

        # 각도 에러 계산
        angle_to_target = self.calculate_angle_to_target()
        angle_error = angle_to_target - self.current_theta
        angle_error = math.atan2(math.sin(angle_error), math.cos(angle_error))  # [-pi, pi]로 정규화

        # 부드러운 속도 감소: 거리 기반 선속도 감소
        max_linear_speed = 7.0  # 선속도의 최대값
        linear_velocity = max(0.2, min(max_linear_speed, distance * 0.5))  # 거리와 비례하여 선속도 감소

        # 부드러운 각속도 감소
        max_angular_speed = 2.0  # 각속도의 최대값
        angular_velocity = max(-max_angular_speed, min(max_angular_speed, angle_error * 2))  # 각도 에러에 비례

        # Twist 메시지 생성
        twist = Twist()
        twist.linear.x = linear_velocity
        twist.angular.z = angular_velocity

        # /cmd_vel 토픽으로 속도 명령 발행
        self.publisher.publish(twist)

def main(args=None):
    rclpy.init(args=args)
    node = MoveToTarget()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
