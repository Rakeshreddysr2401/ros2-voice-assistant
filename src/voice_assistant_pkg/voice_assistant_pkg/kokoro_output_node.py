#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import os
import subprocess
import tempfile
from pathlib import Path
import threading
import queue

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

        # Threading for non-blocking TTS
        self.tts_queue = queue.Queue()
        self.is_speaking = False
        self.tts_thread = None

        # Check if Kokoro is available
        if not KOKORO_AVAILABLE:
            self.get_logger().error(
                "Kokoro TTS not available. Please install: "
                "pip3 install kokoro-onnx soundfile --break-system-packages"
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
                    f"Model files not found at:\n{model_path}\n{voices_path}"
                )
                self.kokoro = None
            else:
                try:
                    self.get_logger().info("⏳ Loading Kokoro TTS model...")
                    self.kokoro = Kokoro(model_path, voices_path)
                    self.get_logger().info("🎤 Kokoro TTS loaded!")

                    # Warmup: Generate a short sample to initialize everything
                    self.get_logger().info("🔥 Warming up model...")
                    _ = self.kokoro.create("Hi", voice="af_sarah", speed=1.0, lang="en-us")
                    self.get_logger().info("✅ Warmup complete!")

                except Exception as e:
                    self.get_logger().error(f"Failed to initialize Kokoro: {e}")
                    self.kokoro = None

        # TTS settings - OPTIMIZED FOR SPEED
        self.voice = "af_sarah"  # Options: af_bella, af_heart, af_sarah, af_sky, am_adam, am_michael
        self.speed = 1.5  # 1.5x speed = 50% faster speech (adjust between 1.2-2.0)
        self.lang = "en-us"

        self.get_logger().info(
            f"📢 OutputNode ready (voice: {self.voice}, speed: {self.speed}x)"
        )

    def handle_response(self, msg: String):
        text = msg.data.strip()
        if not text:
            return

        # Print immediately
        print(f"\n🤖 Assistant says: {text}\n")

        if self.kokoro is None:
            self.get_logger().warn("Kokoro not available, skipping TTS")
            self.publish_done()
            return

        # Start TTS in background thread for non-blocking operation
        if self.tts_thread and self.tts_thread.is_alive():
            self.get_logger().warn("⏭️  Already speaking, queuing...")

        self.tts_thread = threading.Thread(
            target=self._speak_text_threaded,
            args=(text,),
            daemon=True
        )
        self.tts_thread.start()

    def _speak_text_threaded(self, text: str):
        """Run TTS in background thread"""
        try:
            self.is_speaking = True

            # Suspend microphone
            os.system("pactl suspend-source @DEFAULT_SOURCE@ 1 2>/dev/null")

            # Measure generation time
            import time
            start = time.time()

            # Generate audio
            samples, sample_rate = self.kokoro.create(
                text,
                voice=self.voice,
                speed=self.speed,
                lang=self.lang
            )

            gen_time = time.time() - start
            self.get_logger().info(f"⚡ Generated in {gen_time:.2f}s")

            # Save and play
            self.play_audio(samples, sample_rate)

        except Exception as e:
            self.get_logger().error(f"TTS Error: {e}")
        finally:
            # Resume microphone
            os.system("pactl suspend-source @DEFAULT_SOURCE@ 0 2>/dev/null")
            self.is_speaking = False
            self.publish_done()

    def play_audio(self, samples, sample_rate):
        """Save to temp file and play using aplay"""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            tmp_path = tmp_file.name
            sf.write(tmp_path, samples, sample_rate)

        try:
            # Play with aplay - simple and reliable
            subprocess.run(
                ["aplay", "-q", tmp_path],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=30  # Prevent hanging
            )
        except subprocess.TimeoutExpired:
            self.get_logger().error("Audio playback timeout")
        except Exception as e:
            self.get_logger().error(f"Playback error: {e}")
        finally:
            # Cleanup
            try:
                os.unlink(tmp_path)
            except:
                pass

    def publish_done(self):
        """Notify that speaking is complete"""
        done = String()
        done.data = "speaking_done"
        self.status_pub.publish(done)


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

