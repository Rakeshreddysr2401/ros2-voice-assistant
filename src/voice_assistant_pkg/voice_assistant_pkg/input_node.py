#!/usr/bin/env python3
import os, json, queue
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from voice_assistant_pkg.msg_types import VoiceAssistantMsg
import sounddevice as sd
from vosk import Model, KaldiRecognizer

class InputNode(Node):
    """
    Voice -> Text using Vosk (local STT)
    - Uses the system's *default* PulseAudio/PipeWire source for the mic.
    - Keep CPU/RAM low: small Vosk model, 16k mono stream.
    """
    def __init__(self):
        super().__init__('input_node')

        # ---- ROS publisher (same topic/type you already use) ----
        self.pub = self.create_publisher(String, 'user_input', 10)

        # ---- Config (paths, audio) ----
        model_path = os.getenv("VOSK_MODEL_PATH",
                               os.path.expanduser("~/vosk_models/vosk-model-small-en-us-0.15"))
        if not os.path.isdir(model_path):
            self.get_logger().error(
                f"Vosk model not found at {model_path}. "
                f"Set VOSK_MODEL_PATH or install the model.")
            raise SystemExit(1)

        self.sample_rate = 16000  # Vosk likes 8k/16k; 16k is a good default
        self.blocksize = 8000     # ~0.5s blocks at 16k mono int16 (keeps CPU low)

        # ---- Vosk recognizer ----
        self.model = Model(model_path)
        self.rec = KaldiRecognizer(self.model, self.sample_rate)

        # ---- Audio stream + queue ----
        self.q = queue.Queue()

        # Use default PulseAudio/PipeWire source (we set it with pactl earlier)
        # Keeping device=None avoids PortAudio/PulseAudio naming mismatches.
        self.stream = sd.RawInputStream(
            samplerate=self.sample_rate,
            blocksize=self.blocksize,
            dtype='int16',
            channels=1,
            callback=self._audio_cb
        )

        self.stream.start()
        self.timer = self.create_timer(0.05, self._drain_audio)

        self.get_logger().info("🎤 InputNode: listening on DEFAULT source (set via `pactl set-default-source ...`).")
        self.get_logger().info("Speak a phrase; full utterances will be published to 'user_input'.")

    # -------- Audio plumbing --------
    def _audio_cb(self, indata, frames, t, status):
        if status:
            # non-fatal XRUNs etc.
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
                        msg = VoiceAssistantMsg.create_input_msg(text, "voice")
                        self.pub.publish(msg)
                        self.get_logger().info(f"🗣️  {text}")
                # else: partial result available via self.rec.PartialResult(), ignored to keep it simple
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
        try:
            node.stream.stop()
        except Exception:
            pass
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
