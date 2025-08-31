# This is a Python file that defines how to launch multiple ROS2 nodes
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import ExecuteProcess
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    """
    Launch file for the voice assistant system.
    This Python function returns a LaunchDescription that ROS2 uses
    to start all our nodes simultaneously.
    """

    return LaunchDescription([
        # Input Node - handles user input (text/voice)
        Node(
            package='voice_assistant_pkg',
            executable='input_node',
            name='input_node',
            output='screen',
            parameters=[
                {'input_method': 'text'}  # Can be changed to 'voice' later
            ]
        ),

        # Agent Node - processes input with AI
        Node(
            package='voice_assistant_pkg',
            executable='agent_node',
            name='agent_node',
            output='screen',
            parameters=[
                {'agent_type': 'langchain'}
            ]
        ),

        # Output Node - handles responses (print/speak)
        Node(
            package='voice_assistant_pkg',
            executable='output_node',
            name='output_node',
            output='screen',
            parameters=[
                {'output_method': 'text'}  # Can be changed to 'voice' later
            ]
        ),
    ])