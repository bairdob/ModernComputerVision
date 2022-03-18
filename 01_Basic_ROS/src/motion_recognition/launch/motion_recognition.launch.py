from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription(
        [
            Node(
                package="usb_cam",
                executable="usb_cam_node_exe",
                name="camera_publisher",
                parameters=[{"framerate": 5.0, "frame_id": "map"}],
            ),
            Node(
                package="motion_recognition",
                executable="motion_recognition",
                name="motion_recognition",
            ),
            Node(
                package="rviz2",
                executable="rviz2",
                name="rviz",
                arguments=["-d", "/src/motion_recognition/rviz/default.rviz"],
            ),
        ]
    )
