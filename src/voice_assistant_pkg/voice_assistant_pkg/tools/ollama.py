# tools/ollama_tool.py
import requests, base64

MAC = "192.168.1.22:11434"  # your Mac Mini IP

def ollama_query(prompt, image_path=None, model="qwen2.5vl:7b"):
    messages = [{"role": "user", "content": prompt}]
    if image_path:
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        messages[0]["images"] = [b64]

    payload = {"model": model, "messages": messages, "stream": False}
    r = requests.post(f"http://{MAC}/api/chat", json=payload, timeout=120)
    r.raise_for_status()
    data = r.json()
    return data["message"]["content"] if "message" in data else str(data)
