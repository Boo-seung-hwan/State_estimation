from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution, TextSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    world_path = PathJoinSubstitution([
        FindPackageShare("balance_robot_gazebo"),
        "worlds",
        "empty_world.sdf",
    ])

    model_path = PathJoinSubstitution([
        FindPackageShare("balance_robot_description"),
        "models",
        "balance_robot",
        "model.sdf",
    ])

    gz_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            FindPackageShare("ros_gz_sim"),
            "/launch/gz_sim.launch.py",
        ]),
        launch_arguments={
            "gz_args": [TextSubstitution(text="-r -v 4 "), world_path],
        }.items(),
    )

    spawn_robot = Node(
        package="ros_gz_sim",
        executable="create",
        arguments=[
            "-file", model_path,
            "-name", "balance_robot",
            "-x", "0.0",
            "-y", "0.0",
            "-z", "0.15",
        ],
        output="screen",
    )

    return LaunchDescription([
        gz_sim,
        spawn_robot,
    ])
