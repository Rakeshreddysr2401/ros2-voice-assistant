#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import os
import subprocess
import wave
from pathlib import Path

try:
    from piper import PiperVoice

    PIPER_AVAILABLE = True
except ImportError:
    PIPER_AVAILABLE = False


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
        self.is_speaking = False

        # Check if Piper is available
        if not PIPER_AVAILABLE:
            self.get_logger().error(
                "Piper TTS not available. Please install: "
                "pip3 install piper-tts --break-system-packages"
            )
            self.voice = None
        else:
            # Initialize Piper with model files
            src_dir = os.path.expanduser("~/ros2_ws/src/voice_assistant_pkg/voice_assistant_pkg")
            model_path = os.path.join(src_dir, "piper_models", "en_US-lessac-low.onnx")

            # Check if model file exists
            if not Path(model_path).exists():
                self.get_logger().error(
                    f"Model file not found at: {model_path}\n"
                    f"Download it with:\n"
                    f"mkdir -p {os.path.dirname(model_path)}\n"
                    f"cd {os.path.dirname(model_path)}\n"
                    f"wget https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/low/en_US-lessac-low.onnx\n"
                    f"wget https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/low/en_US-lessac-low.onnx.json"
                )
                self.voice = None
            else:
                try:
                    self.get_logger().info("Loading Piper TTS model...")
                    self.voice = PiperVoice.load(model_path)
                    self.get_logger().info("🎤 Piper TTS initialized successfully!")

                    # Warmup
                    self.get_logger().info("Warming up TTS...")
                    list(self.voice.synthesize("Hello"))
                    self.get_logger().info("✅ Warmup complete!")

                except Exception as e:
                    self.get_logger().error(f"Failed to initialize Piper: {e}")
                    self.voice = None

        self.get_logger().info("📢 OutputNode started with Piper TTS")

    def handle_response(self, msg: String):
        text = msg.data.strip()
        if not text:
            return

        # Don't queue new speech if already speaking
        if self.is_speaking:
            self.get_logger().warn("Already speaking, skipping...")
            return

        # Print the assistant response to the console
        print(f"\n🤖 Assistant says: {text}\n")

        # Suspend microphone before speaking
        os.system("pactl suspend-source @DEFAULT_SOURCE@ 1")

        # Generate and play speech
        if self.voice is not None:
            try:
                self.is_speaking = True
                start = self.get_clock().now()
                self.speak_text(text)
                elapsed = (self.get_clock().now() - start).nanoseconds / 1e9
                self.get_logger().info(f"⏱️  Total time: {elapsed:.2f}s")
            except Exception as e:
                self.get_logger().error(f"TTS Error: {e}")
            finally:
                self.is_speaking = False
        else:
            self.get_logger().warn("Piper not available, skipping TTS")

        # Resume microphone
        os.system("pactl suspend-source @DEFAULT_SOURCE@ 0")

        # Notify input_node that "speaking" is done
        done = String()
        done.data = "speaking_done"
        self.status_pub.publish(done)

    def speak_text(self, text: str):
        """Generate speech and play it using aplay"""
        import tempfile

        # Create temporary WAV file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            tmp_path = tmp_file.name

        try:
            # Generate audio to file
            with wave.open(tmp_path, "wb") as wav_file:
                wav_file.setnchannels(1)  # Mono
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(22050)  # 22050 Hz sample rate
                self.voice.synthesize(text, wav_file)

            # Play audio using aplay
            result = subprocess.run(
                ["aplay", "-q", tmp_path],
                capture_output=True,
                text=True
            )

            if result.returncode != 0:
                self.get_logger().error(f"aplay failed: {result.stderr}")

        except Exception as e:
            self.get_logger().error(f"TTS Error: {e}")
        finally:
            # Clean up temp file
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