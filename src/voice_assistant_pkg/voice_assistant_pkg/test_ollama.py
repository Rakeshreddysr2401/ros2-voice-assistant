# test_ollama.py
from tools.ollama import ollama_query


image_path = "road.jpg"
print("=== TEXT TEST ===")
resp = ollama_query("What you are able to see in diamond shop?",image_path=image_path)
print(resp)
