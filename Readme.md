colcon build --packages-select voice_assistant_pkg

source install/setup.bash

AUDIO_DEVICE_INDEX=0 ros2 run voice_assistant_pkg input_node

 ros2 run voice_assistant_pkg agent_node

CAMERA_SOURCE=/dev/video0 ros2 run voice_assistant_pkg camera_publisher_node

ros2 run voice_assistant_pkg blip_server

ros2 run voice_assistant_pkg yolo_server

ros2 run voice_assistant_pkg output_node

ros2 run voice_assistant_pkg qwen_vision_server




source /opt/ros/jazzy/setup.bash
source ~/microros_ws/install/setup.bash
source ~/ros2_ws/install/setup.bash




source /opt/ros/jazzy/setup.bash
source ~/microros_ws/install/local_setup.bash   # if you built the agent here; else skip if installed by snap
ros2 run micro_ros_agent micro_ros_agent udp4 --port 8888


source /opt/ros/jazzy/setup.bash
source ~/ros2_ws/install/setup.bash
ros2 run voice_assistant_pkg servo_publisher
