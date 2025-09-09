#!/usr/bin/env python3
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
    """Voice -> Text using tiny Whisper with lightweight CPU optimizations"""

    def __init__(self):
        super().__init__('input_node')
        self.pub = self.create_publisher(String, 'user_input', 10)

        # Whisper model (tiny/int8 = fast on Pi5)
        model_size = os.getenv("WHISPER_MODEL_SIZE", "tiny")
        self.model = WhisperModel(
            model_size,
            device="cpu",
            compute_type="int8",
            num_workers=1,
            cpu_threads=2
        )

        self.sample_rate = 16000
        self.blocksize = 3200  # ~0.2s chunks

        # Short buffer, fast turnaround
        self.audio_buffer = deque(maxlen=48000)  # ~3s
        self.min_audio_length = 8000  # 0.5s
        self.silence_threshold = 0.8  # End of phrase

        # Mic input queue
        self.q = queue.Queue(maxsize=10)
        self.stream = sd.RawInputStream(
            samplerate=self.sample_rate,
            blocksize=self.blocksize,
            dtype='int16',
            channels=1,
            callback=self._audio_cb,
        )
        self.stream.start()

        # VAD (voice activity detection)
        self.vad = webrtcvad.Vad(2)  # balanced

        # Tracking speech
        self.last_speech_time = 0
        self.is_recording = False
        self.speech_frames = []
        self.processing = False

        # Run processing loop every 200ms
        self.timer = self.create_timer(0.2, self._process_audio)

        # Background transcription thread
        self.transcription_queue = queue.Queue(maxsize=2)
        self.transcription_thread = threading.Thread(
            target=self._transcription_worker,
            daemon=True
        )
        self.transcription_thread.start()

        self.get_logger().info(f"🎤 InputNode ready (Whisper {model_size}, CPU optimized)")

    def _audio_cb(self, indata, frames, t, status):
        if status:
            self.get_logger().warn(str(status))
        if not self.q.full():
            self.q.put(np.frombuffer(indata, dtype=np.int16).copy())

    def _is_speech_simple(self, audio_chunk):
        """Fast speech detection using VAD"""
        audio_bytes = (audio_chunk * 32767).astype(np.int16).tobytes()
        frame_size = 640  # 20ms @16kHz
        if len(audio_bytes) >= frame_size:
            try:
                return self.vad.is_speech(audio_bytes[:frame_size], self.sample_rate)
            except:
                return False
        return False

    def _process_audio(self):
        if self.processing:
            return

        audio_chunks = []
        try:
            for _ in range(5):  # limit per cycle
                audio_chunks.append(self.q.get_nowait())
        except queue.Empty:
            pass

        if not audio_chunks:
            return

        audio_data = np.concatenate(audio_chunks).astype(np.float32) / 32768.0
        current_time = time.time()

        # Quick RMS check to skip silence
        rms = np.sqrt(np.mean(audio_data ** 2))
        has_speech = rms >= 0.001 and self._is_speech_simple(audio_data)

        if has_speech:
            self.last_speech_time = current_time
            if not self.is_recording:
                self.is_recording = True
                self.speech_frames = []

        if self.is_recording:
            self.speech_frames.extend(audio_data)

            if (current_time - self.last_speech_time > self.silence_threshold
                    and len(self.speech_frames) >= self.min_audio_length):
                if not self.transcription_queue.full():
                    self.transcription_queue.put(self.speech_frames.copy())
                self.speech_frames = []
                self.is_recording = False

    def _transcription_worker(self):
        while True:
            try:
                speech_data = self.transcription_queue.get(timeout=1.0)
                self._transcribe_speech(speech_data)
            except queue.Empty:
                continue
            except Exception as e:
                self.get_logger().error(f"Transcription worker error: {e}")

    def _transcribe_speech(self, speech_data):
        if not speech_data:
            return
        try:
            self.processing = True
            audio_np = np.array(speech_data, dtype=np.float32)

            # Fast transcription settings
            segments, _ = self.model.transcribe(
                audio_np,
                beam_size=1,
                best_of=1,
                language="en",
                temperature=0.0,
                condition_on_previous_text=False,
                no_speech_threshold=0.6,
            )

            text_parts = [seg.text.strip() for seg in segments if seg.text.strip()]
            if text_parts:
                final_text = " ".join(" ".join(text_parts).split())
                if final_text:
                    msg = String()
                    msg.data = final_text
                    self.pub.publish(msg)
                    self.get_logger().info(f"🗣️ {final_text}")

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
