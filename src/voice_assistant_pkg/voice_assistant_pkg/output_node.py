#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import subprocess
import numpy as np
import queue
import threading
import os
import time
from piper.voice import PiperVoice


class PiperStreamingNode(Node):
    def __init__(self):
        super().__init__("piper_streaming_node")

        # ----------------------------------------------------------
        # Load Piper TTS
        # ----------------------------------------------------------
        model_path = os.path.expanduser("~/piper_models/en_US-amy-medium.onnx")
        config_path = model_path + ".json"

        self.get_logger().info("⏳ Loading Piper voice...")
        self.voice = PiperVoice.load(model_path, config_path)
        self.get_logger().info("🎤 Piper model loaded successfully!")

        # ----------------------------------------------------------
        # ROS interfaces
        # ----------------------------------------------------------
        self.subscription = self.create_subscription(
            String, "agent_response", self.handle_response, 10
        )
        self.status_pub = self.create_publisher(String, "output_status", 10)

        # Background worker
        self.tts_queue = queue.Queue()
        self.worker_shutdown = threading.Event()
        threading.Thread(target=self._worker_loop, daemon=True).start()

        self.get_logger().info("🚀 PiperStreamingNode ready!")

    # ----------------------------------------------------------
    # Incoming messages
    # ----------------------------------------------------------
    def handle_response(self, msg: String):
        text = msg.data.strip()
        if not text:
            return

        print(f"\n🤖 Assistant says: {text}\n")
        self.get_logger().info(f"📝 Queueing text: \"{text}\"")
        self.tts_queue.put(text)

    # ----------------------------------------------------------
    # Background TTS Worker
    # ----------------------------------------------------------
    def _worker_loop(self):
        while not self.worker_shutdown.is_set():

            try:
                text = self.tts_queue.get(timeout=0.25)
            except queue.Empty:
                continue

            self.get_logger().info(f"🔊 Starting TTS for: \"{text}\"")

            # Mute microphone during playback
            os.system("pactl set-source-mute @DEFAULT_SOURCE@ 1 2>/dev/null")

            try:
                # ------------------------------------------------------
                # Piper streaming generator
                # ------------------------------------------------------
                chunk_gen = self.voice.synthesize(text)

                # Get first chunk
                try:
                    first_chunk = next(chunk_gen)
                except StopIteration:
                    self.get_logger().error("❌ No audio returned!")
                    self._speak_done()
                    continue

                # Extract correct fields
                sr = first_chunk.sample_rate
                audio_f32 = first_chunk.audio_float_array

                if audio_f32 is None:
                    self.get_logger().error("❌ First chunk audio_float_array is None")
                    self._speak_done()
                    continue

                # Convert float32 → int16
                pcm16 = (audio_f32 * 32767).astype(np.int16)

                # ------------------------------------------------------
                # Launch ffplay
                # ------------------------------------------------------
                ffplay_cmd = [
                    "ffplay",
                    "-f", "s16le",
                    "-ar", str(sr),
                    "-ac", "1",
                    "-nodisp",
                    "-autoexit",
                    "-hide_banner",
                    "-loglevel", "quiet",
                    "-i", "pipe:0"
                ]

                proc = subprocess.Popen(ffplay_cmd, stdin=subprocess.PIPE)
                proc.stdin.write(pcm16.tobytes())
                proc.stdin.flush()

                chunk_count = 1

                # ------------------------------------------------------
                # Stream the remaining chunks
                # ------------------------------------------------------
                for chunk in chunk_gen:
                    audio_f32 = chunk.audio_float_array
                    if audio_f32 is None:
                        continue

                    pcm16 = (audio_f32 * 32767).astype(np.int16)
                    proc.stdin.write(pcm16.tobytes())
                    proc.stdin.flush()
                    chunk_count += 1

                # Close playback
                try:
                    proc.stdin.close()
                except:
                    pass
                proc.wait()

                self.get_logger().info(f"🔈 Streamed {chunk_count} audio chunks")

            except Exception as e:
                self.get_logger().error(f"❌ TTS error: {e}")

            finally:
                self._speak_done()
                self.tts_queue.task_done()

    # ----------------------------------------------------------
    # Publish done + unmute mic
    # ----------------------------------------------------------
    def _speak_done(self):
        os.system("pactl set-source-mute @DEFAULT_SOURCE@ 0 2>/dev/null")
        self.status_pub.publish(String(data="speaking_done"))
        self.get_logger().info("🟢 Speaking done")

    # ----------------------------------------------------------
    # Shutdown cleanly
    # ----------------------------------------------------------
    def destroy_node(self):
        self.worker_shutdown.set()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = PiperStreamingNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
