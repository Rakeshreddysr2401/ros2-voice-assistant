#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import os
import subprocess
import tempfile
from pathlib import Path

try:
    import soundfile as sf
    from kokoro_onnx import Kokoro

    KOKORO_AVAILABLE = True
except ImportError:
    KOKORO_AVAILABLE = False


class OutputNode(Node):
    def __init__(self):
        super().__init__("output_node")

        self.subscription = self.create_subscription(
            String,
            "agent_response",
            self.handle_response,
            10
        )

        self.status_pub = self.create_publisher(String, "output_status", 10)

        # Check if Kokoro is available
        if not KOKORO_AVAILABLE:
            self.get_logger().error(
                "Kokoro TTS not available. Please install: "
                "pip install kokoro-onnx soundfile"
            )
            self.kokoro = None
        else:

            # Initialize Kokoro with model files (using source directory)
            src_dir = os.path.expanduser("~/ros2_ws/src/voice_assistant_pkg/voice_assistant_pkg")
            model_path = os.path.join(src_dir, "models", "kokoro-v1.0.int8.onnx")
            voices_path = os.path.join(src_dir, "models", "voices-v1.0.bin")

            # Check if model files exist
            if not Path(model_path).exists() or not Path(voices_path).exists():
                self.get_logger().error(
                    f"Model files not found!\n"
                    f"Please download them:\n"
                    f"wget https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.int8.onnx\n"
                    f"wget https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin"
                )
                self.kokoro = None
            else:
                try:
                    self.kokoro = Kokoro(model_path, voices_path)
                    self.get_logger().info("🎤 Kokoro TTS initialized successfully!")
                except Exception as e:
                    self.get_logger().error(f"Failed to initialize Kokoro: {e}")
                    self.kokoro = None

        # TTS settings - configurable
        self.voice = "af_sarah"  # Available: af_bella, af_heart, af_sarah, af_sky, etc.
        self.speed = 1.0
        self.lang = "en-us"

        self.get_logger().info(
            f"📢 OutputNode started with Kokoro TTS (voice: {self.voice}, speed: {self.speed})"
        )

    def handle_response(self, msg: String):
        text = msg.data.strip()
        if not text:
            return

        # Print the assistant response to the console
        print(f"\n🤖 Assistant says: {text}\n")

        # Suspend microphone before speaking
        os.system("pactl suspend-source @DEFAULT_SOURCE@ 1")

        # Generate and play speech
        if self.kokoro is not None:
            try:
                self.speak_text(text)
            except Exception as e:
                self.get_logger().error(f"TTS Error: {e}")
        else:
            self.get_logger().warn("Kokoro not available, skipping TTS")

        # Resume microphone
        os.system("pactl suspend-source @DEFAULT_SOURCE@ 0")

        # Notify input_node that "speaking" is done
        done = String()
        done.data = "speaking_done"
        self.status_pub.publish(done)

    def speak_text(self, text: str):
        """Generate speech and play it using aplay"""
        # Generate audio samples
        samples, sample_rate = self.kokoro.create(
            text,
            voice=self.voice,
            speed=self.speed,
            lang=self.lang
        )

        # Save to temporary file and play
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            tmp_path = tmp_file.name
            sf.write(tmp_path, samples, sample_rate)

        try:
            # Play audio using aplay (pre-installed on most Pi systems)
            subprocess.run(
                ["aplay", "-q", tmp_path],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
        finally:
            # Clean up temporary file
            try:
                os.unlink(tmp_path)
            except:
                pass


def main(args=None):
    rclpy.init(args=args)
    node = OutputNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()