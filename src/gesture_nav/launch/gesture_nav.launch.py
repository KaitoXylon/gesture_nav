import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('camera_index', default_value='0', description='Webcam device index'),
        DeclareLaunchArgument('model_path', default_value='yolo26n.pt', description='YOLO model weights'),
        DeclareLaunchArgument('target_distance', default_value='3.0', description='Target distance to maintain (m)'),
        DeclareLaunchArgument('show_preview', default_value='True', description='Show OpenCV preview windows'),

        # 1. MediaPipe Webcam Hand Gesture Node
        Node(
            package='gesture_nav',
            executable='webcam_gesture_node',
            name='webcam_gesture_node',
            output='screen',
            parameters=[{
                'camera_index': LaunchConfiguration('camera_index'),
                'show_preview': LaunchConfiguration('show_preview'),
                'debounce_frames': 6,
                'topic_name': '/rover/gesture_state',
            }]
        ),

        # 2. YOLO Human Detection & Follower Node
        Node(
            package='gesture_nav',
            executable='human_follower_node',
            name='human_follower_node',
            output='screen',
            parameters=[{
                'model_path': LaunchConfiguration('model_path'),
                'camera_topic': '/zed2i/zed_node/left/image_rect_color',
                'cmd_vel_topic': '/cmd_vel',
                'gesture_topic': '/rover/gesture_state',
                'target_distance_m': LaunchConfiguration('target_distance'),
                'show_preview': LaunchConfiguration('show_preview'),
            }]
        )
    ])
