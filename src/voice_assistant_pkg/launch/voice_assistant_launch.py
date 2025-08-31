import os
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import pyttsx3
import sounddevice as sd
import soundfile as sf
import tempfile
import subprocess


def get_bluetooth_output_device():
    """Find bluetooth sink for output (buds speaker)."""
    try:
        result = subprocess.check_output(
            ["pactl", "list", "short", "sinks"], text=True
        )
        for line in result.splitlines():
            if "bluez_sink" in line:  # Bluetooth speaker
                return line.split()[1]
    except Exception as e:
        print(f"[OutputNode] Could not detect Bluetooth output: {e}")
    return None


class OutputNode(Node):
    def __init__(self):
        super().__init__('output_node')

        # Subscribe to the agent response topic
        self.subscription = self.create_subscription(
            String,
            'agent_response',   # 👈 matches the topic name published by your agent
            self.speak_callback,
            10
        )

        # Detect Bluetooth sink (buds speaker)
        self.output_device = os.getenv("VOICE_OUTPUT_DEVICE", None)
        if not self.output_device:
            self.output_device = get_bluetooth_output_device()

        self.get_logger().info(
            f"🎧 Output device: {self.output_device or 'Default system speaker'}"
        )

    def speak_callback(self, msg):
        text = msg.data
        self.get_logger().info(f"🗣 Speaking: {text}")

        try:
            # Convert text to speech (save as temporary file)
            engine = pyttsx3.init()
            tmpfile = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            engine.save_to_file(text, tmpfile.name)
            engine.runAndWait()

            # Play through sounddevice (force Bluetooth sink if available)
            data, fs = sf.read(tmpfile.name, dtype='float32')
            if self.output_device:
                sd.play(data, fs, device=self.output_device)
            else:
                sd.play(data, fs)  # default speaker
            sd.wait()

        except Exception as e:
            self.get_logger().error(f"❌ Error in speech synthesis: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = OutputNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
