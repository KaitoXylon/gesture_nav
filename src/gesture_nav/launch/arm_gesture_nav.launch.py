# Starts the arm gesture rover node using YOLO26n-pose (opens ZED 2i directly via pyzed)
# Example: ros2 launch gesture_nav arm_gesture_nav.launch.py target_distance:=2.0

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    target_distance = LaunchConfiguration('target_distance')
    show_preview = LaunchConfiguration('show_preview')
    camera_x = LaunchConfiguration('camera_x')
    camera_z = LaunchConfiguration('camera_z')
    device = LaunchConfiguration('device')
    human_height = LaunchConfiguration('human_height')
    focal_length = LaunchConfiguration('focal_length')

    return LaunchDescription([
        DeclareLaunchArgument('target_distance', default_value='1.0'),
        DeclareLaunchArgument('show_preview', default_value='true'),
        # Camera position on the rover, from base_link (meters): x forward, z up
        DeclareLaunchArgument('camera_x', default_value='0.0'),
        DeclareLaunchArgument('camera_z', default_value='0.0'),
        DeclareLaunchArgument('device', default_value='cpu'),
        DeclareLaunchArgument('human_height', default_value='1.75'),
        DeclareLaunchArgument('focal_length', default_value='0.0'),

        Node(
            package='gesture_nav',
            executable='arm_gesture_node',
            output='screen',
            emulate_tty=True,  # print() lines show up right away
            parameters=[{
                'target_distance': ParameterValue(target_distance, value_type=float),
                'show_preview': show_preview,
                'camera_x': ParameterValue(camera_x, value_type=float),
                'camera_z': ParameterValue(camera_z, value_type=float),
                'device': device,
                'human_height': ParameterValue(human_height, value_type=float),
                'focal_length': ParameterValue(focal_length, value_type=float),
            }],
        ),
    ])
