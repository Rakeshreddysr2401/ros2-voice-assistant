import os
import queue
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import sounddevice as sd
import numpy as np
import webrtcvad
from faster_whisper import WhisperModel
import time
from collections import deque
import threading


class InputNode(Node):
    """Lightweight Voice -> Text using tiny Whisper model with CPU optimizations"""

    def __init__(self):
        super().__init__('input_node')
        self.pub = self.create_publisher(String, 'user_input', 10)

        # Use tiny model but with optimizations for better accuracy
        model_size = os.getenv("WHISPER_MODEL_SIZE", "tiny")

        # Use int8 quantization for lowest CPU usage
        self.model = WhisperModel(
            model_size,
            device="cpu",
            compute_type="int8",
            num_workers=1,
            cpu_threads=2  # Limit CPU threads
        )

        self.sample_rate = 16000
        self.blocksize = 3200  # Smaller chunks to reduce processing load

        # Minimal buffering to reduce memory and CPU
        self.audio_buffer = deque(maxlen=48000)  # 3 seconds max
        self.min_audio_length = 8000  # 0.5 seconds minimum
        self.silence_threshold = 0.8  # Wait longer for complete phrases

        # Setup queue + mic stream
        self.q = queue.Queue(maxsize=10)  # Limit queue size
        self.stream = sd.RawInputStream(
            samplerate=self.sample_rate,
            blocksize=self.blocksize,
            dtype='int16',
            channels=1,
            callback=self._audio_cb,
        )
        self.stream.start()

        # Less aggressive VAD to reduce false positives
        self.vad = webrtcvad.Vad(2)  # Balanced setting

        # Tracking variables
        self.last_speech_time = 0
        self.is_recording = False
        self.speech_frames = []
        self.processing = False  # Prevent overlapping processing

        # Process less frequently to reduce CPU load
        self.timer = self.create_timer(0.2, self._process_audio)

        # Background thread for transcription to avoid blocking
        self.transcription_queue = queue.Queue(maxsize=2)
        self.transcription_thread = threading.Thread(target=self._transcription_worker, daemon=True)
        self.transcription_thread.start()

        self.get_logger().info(f"🎤 Lightweight InputNode: Whisper ({model_size}) ready (CPU optimized)")

    def _audio_cb(self, indata, frames, t, status):
        if status:
            self.get_logger().warn(str(status))

        # Drop audio if queue is full (prevents buildup)
        if not self.q.full():
            audio_data = np.frombuffer(indata, dtype=np.int16)
            self.q.put(audio_data.copy())

    def _is_speech_simple(self, audio_chunk):
        """Simplified speech detection to reduce CPU load"""
        # Convert to bytes for VAD
        audio_bytes = (audio_chunk * 32767).astype(np.int16).tobytes()

        # Use only one frame check to reduce processing
        frame_size = 640  # 20ms at 16kHz = 320 samples * 2 bytes
        if len(audio_bytes) >= frame_size:
            try:
                return self.vad.is_speech(audio_bytes[:frame_size], self.sample_rate)
            except:
                return False
        return False

    def _process_audio(self):
        if self.processing:  # Skip if already processing
            return

        # Collect limited audio to reduce memory usage
        audio_chunks = []
        chunks_collected = 0
        max_chunks = 5  # Limit processing per cycle

        try:
            while chunks_collected < max_chunks:
                chunk = self.q.get_nowait()
                audio_chunks.append(chunk)
                chunks_collected += 1
        except queue.Empty:
            pass

        if not audio_chunks:
            return

        # Quick processing
        audio_data = np.concatenate(audio_chunks).astype(np.float32) / 32768.0
        current_time = time.time()

        # Simple volume-based pre-filter to reduce VAD calls
        rms = np.sqrt(np.mean(audio_data ** 2))
        if rms < 0.001:  # Very quiet, skip VAD
            has_speech = False
        else:
            has_speech = self._is_speech_simple(audio_data)

        if has_speech:
            self.last_speech_time = current_time
            if not self.is_recording:
                self.is_recording = True
                self.speech_frames = []

        # Collect speech
        if self.is_recording:
            self.speech_frames.extend(audio_data)

            # Stop recording and queue for transcription
            if (current_time - self.last_speech_time > self.silence_threshold and
                    len(self.speech_frames) >= self.min_audio_length):

                # Queue for background transcription
                if not self.transcription_queue.full():
                    self.transcription_queue.put(self.speech_frames.copy())

                self.speech_frames = []
                self.is_recording = False

    def _transcription_worker(self):
        """Background thread for transcription"""
        while True:
            try:
                speech_data = self.transcription_queue.get(timeout=1.0)
                self._transcribe_speech(speech_data)
            except queue.Empty:
                continue
            except Exception as e:
                self.get_logger().error(f"Transcription worker error: {e}")

    def _transcribe_speech(self, speech_data):
        """Lightweight transcription with minimal processing"""
        if not speech_data:
            return

        try:
            self.processing = True
            audio_np = np.array(speech_data, dtype=np.float32)

            # Ultra-light Whisper settings for speed
            segments, info = self.model.transcribe(
                audio_np,
                beam_size=1,  # Minimal beam search
                best_of=1,  # Single candidate
                language="en",  # Fixed language
                temperature=0.0,  # Deterministic
                condition_on_previous_text=False,
                no_speech_threshold=0.6,
                # Remove other heavy processing options
            )

            # Quick text extraction
            text_parts = []
            for segment in segments:
                text = segment.text.strip()
                if text and len(text) > 1:
                    text_parts.append(text)

            if text_parts:
                final_text = " ".join(text_parts)
                # Minimal cleaning
                final_text = " ".join(final_text.split())  # Clean whitespace

                if final_text:
                    msg = String()
                    msg.data = final_text
                    self.pub.publish(msg)
                    self.get_logger().info(f"🗣️ '{final_text}'")

        except Exception as e:
            self.get_logger().error(f"❌ Transcription error: {e}")
        finally:
            self.processing = False


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


if __name__ == "__main__":
    main()


