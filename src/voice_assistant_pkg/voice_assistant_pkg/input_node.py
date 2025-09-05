# input_node.py
#!/usr/bin/env python3
import os
import json
import queue
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import sounddevice as sd
from vosk import Model, KaldiRecognizer

class InputNode(Node):
    """Voice -> Text using Vosk (local STT)"""

    def __init__(self):
        super().__init__('input_node')
        self.pub = self.create_publisher(String, 'user_input', 10)

        model_path = os.getenv("VOSK_MODEL_PATH",
                               os.path.expanduser("~/vosk_models/vosk-model-small-en-us-0.15"))
        if not os.path.isdir(model_path):
            self.get_logger().error(f"Vosk model not found at {model_path}")
            raise SystemExit(1)

        self.sample_rate = 16000
        self.blocksize = 8000

        self.model = Model(model_path)
        self.rec = KaldiRecognizer(self.model, self.sample_rate)

        self.q = queue.Queue()
        self.stream = sd.RawInputStream(
            samplerate=self.sample_rate,
            blocksize=self.blocksize,
            dtype='int16',
            channels=1,
            callback=self._audio_cb
        )
        self.stream.start()
        self.timer = self.create_timer(0.05, self._drain_audio)
        self.get_logger().info("🎤 InputNode: listening for voice input.")

    def _audio_cb(self, indata, frames, t, status):
        if status:
            self.get_logger().warn(str(status))
        self.q.put(bytes(indata))

    def _drain_audio(self):
        try:
            while True:
                data = self.q.get_nowait()
                if self.rec.AcceptWaveform(data):
                    result = json.loads(self.rec.Result())
                    text = (result.get("text") or "").strip()
                    if text:
                        # Simple string message - just the text
                        msg = String()
                        msg.data = text
                        self.pub.publish(msg)
                        self.get_logger().info(f"🗣️  {text}")
        except queue.Empty:
            pass

def main(args=None):
    rclpy.init(args=args)
    node = InputNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stream.stop()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()