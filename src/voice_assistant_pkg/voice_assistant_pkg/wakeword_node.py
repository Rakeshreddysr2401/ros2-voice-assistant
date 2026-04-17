#!/usr/bin/env python3
import os
import queue
import json
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool
import sounddevice as sd
from vosk import Model, KaldiRecognizer

class WakeWordNode(Node):
    def __init__(self):
        super().__init__('wakeword_node')
        
        # Configuration
        self.wake_word = os.getenv("WAKE_WORD", "jarvis").lower()
        self.pub_wake = self.create_publisher(Bool, 'wake_word_detected', 10)
        self.status_pub = self.create_publisher(String, 'agent_status', 10)
        
        # Audio Settings
        self.sample_rate = 16000
        self.device_index = int(os.getenv("AUDIO_DEVICE_INDEX", "1"))
        self.q = queue.Queue()

        # Load Vosk Model
        # Assumes model is in a standard location or local 'model' folder
        model_path = os.getenv("VOSK_MODEL_PATH", "model-small")
        if not os.path.exists(model_path):
            self.get_logger().error(f"❌ Vosk model not found at {model_path}. Please download 'vosk-model-small-en-us' and rename to 'model-small'.")
            # We'll try to find it in the package dir if not found
            package_path = os.path.expanduser("~/ros2_ws/src/voice_assistant_pkg/voice_assistant_pkg/model-small")
            if os.path.exists(package_path):
                model_path = package_path
            else:
                self.get_logger().error("Could not find Vosk model. Wake word detection will not work.")
                return

        self.model = Model(model_path)
        # We only care about the wake word to save CPU
        self.rec = KaldiRecognizer(self.model, self.sample_rate, f'["{self.wake_word}", "[unk]"]')

        # Start Audio Stream
        self.stream = sd.RawInputStream(
            samplerate=self.sample_rate,
            blocksize=8000,
            dtype='int16',
            channels=1,
            device=self.device_index,
            callback=self._audio_cb
        )
        self.stream.start()
        
        # Timer to process audio
        self.create_timer(0.1, self._process_audio)
        
        self.get_logger().info(f"👂 Listening for wake word: '{self.wake_word}'")

    def _audio_cb(self, indata, frames, t, status):
        self.q.put(bytes(indata))

    def _process_audio(self):
        while not self.q.empty():
            data = self.q.get()
            if self.rec.AcceptWaveform(data):
                result = json.loads(self.rec.Result())
                text = result.get("text", "").lower()
                if self.wake_word in text:
                    self._trigger_wake()
            else:
                # Partial results can also contain the word
                partial = json.loads(self.rec.PartialResult())
                if self.wake_word in partial.get("partial", "").lower():
                    self._trigger_wake()
                    self.rec.Reset() # Clear for next time

    def _trigger_wake(self):
        self.get_logger().info(f"✨ Wake word '{self.wake_word}' detected!")
        msg = Bool()
        msg.data = True
        self.pub_wake.publish(msg)
        
        # Visual feedback on dashboard
        status = String()
        status.data = "Listening..."
        self.status_pub.publish(status)

def main(args=None):
    rclpy.init(args=args)
    node = WakeWordNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
