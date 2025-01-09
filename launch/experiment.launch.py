#!/usr/bin/env python3

import os
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # 경로 가져오기
    moving_people_launch_path = os.path.join(
        get_package_share_directory('moving_people'),
        'launch',
        'special_csv_crowd_people.launch.py'
    )
    
    # flow_nav의 move_to_coordinate.py 노드
    move_to_coordinate_node = Node(
        package='flow_nav',
        executable='move_to_coordinate.py',
        name='move_to_coordinate',
        output='screen',
    )

    # moving_people의 launch 파일 포함
    special_crowd_people_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(moving_people_launch_path)
    )

    # RViz 노드 실행
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', os.path.join(
            get_package_share_directory('flow_nav'),
            'rviz',
            'flow_nav.rviz'
        )]
    )

    return LaunchDescription([
        move_to_coordinate_node,
        special_crowd_people_launch,
        rviz_node,
    ])
