# Starts the rover node (it opens the ZED 2i itself, no ZED ROS wrapper needed).
# Example: ros2 launch gesture_nav gesture_nav.launch.py target_distance:=2.0

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

    return LaunchDescription([
        DeclareLaunchArgument('target_distance', default_value='1.0'),
        DeclareLaunchArgument('show_preview', default_value='true'),
        # Camera position on the rover, from base_link (meters): x forward, z up
        DeclareLaunchArgument('camera_x', default_value='0.0'),
        DeclareLaunchArgument('camera_z', default_value='0.0'),

        Node(
            package='gesture_nav',
            executable='rover_node',
            output='screen',
            emulate_tty=True,  # print() lines show up right away
            parameters=[{
                'target_distance': ParameterValue(target_distance, value_type=float),
                'show_preview': show_preview,
                'camera_x': ParameterValue(camera_x, value_type=float),
                'camera_z': ParameterValue(camera_z, value_type=float),
            }],
        ),
    ])
