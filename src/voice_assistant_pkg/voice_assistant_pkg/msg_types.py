# voice_assistant_pkg/msg_types.py
from std_msgs.msg import String
import json
import time

class VoiceAssistantMsg:
    """Custom message wrapper for voice assistant communication"""

    @staticmethod
    def create_input_msg(text: str, source: str = "text") -> String:
        msg = String()
        msg.data = json.dumps({
            "type": "input",
            "text": text,
            "source": source,
            "timestamp": time.time()
        })
        return msg

    @staticmethod
    def create_response_msg(text: str, metadata: dict = None) -> String:
        msg = String()
        msg.data = json.dumps({
            "type": "response",
            "text": text,
            "metadata": metadata or {},
            "timestamp": time.time()
        })
        return msg

    @staticmethod
    def parse_msg(msg: String) -> dict:
        return json.loads(msg.data)
