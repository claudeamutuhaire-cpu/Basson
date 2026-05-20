import os
import requests
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse, FileResponse
from faster_whisper import WhisperModel
from duckduckgo_search import DDGS
import subprocess
import tempfile

app = FastAPI()

# Config
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.1")
VISION_MODEL = os.getenv("VISION_MODEL", "moondream2")

# Load Whisper once at startup. Use "tiny" for Codespaces, "large-v3" for PC
stt_model = WhisperModel("tiny", device="cpu", compute_type="int8")

@app.post("/chat")
def chat(message: dict):
    """Text chat endpoint. Supports tool call for web search."""
    user_msg = message["message"]

    # Simple tool use: if message starts with /search, do web search
    if user_msg.startswith("/search"):
        query = user_msg.replace("/search", "").strip()
        with DDGS() as ddgs:
            results = [r for r in ddgs.text(query, max_results=3)]
        context = "\n".join([f"{r['title']}: {r['body']}" for r in results])
        user_msg = f"Use this info to answer:\n{context}\n\nQuestion: {query}"

    # Call Ollama
    payload = {
        "model": LLM_MODEL,
        "messages": [{"role": "user", "content": user_msg}],
        "stream": False
    }
    r = requests.post(f"{OLLAMA_URL}/api/chat", json=payload)
    reply = r.json()["message"]["content"]
    return {"reply": reply}

@app.post("/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    """Speech to text using faster-whisper"""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(await audio.read())
        segments, _ = stt_model.transcribe(tmp.name)
        text = " ".join([seg.text for seg in segments])
    return {"text": text}

@app.post("/speak")
def speak(text: dict):
    """Text to speech using Piper. Returns wav file."""
    text_str = text["text"]
    output_path = "/tmp/output.wav"

    # Piper needs to be installed in container. Adjust model path as needed
    cmd = [
        "piper",
        "--model", "/app/piper/en_US-lessac-medium.onnx",
        "--output_file", output_path
    ]
    proc = subprocess.run(cmd, input=text_str.encode(), capture_output=True)

    return FileResponse(output_path, media_type="audio/wav")

@app.post("/vision")
def vision(data: dict):
    """Image + question using Moondream2 via Ollama"""
    image_b64 = data["image"] # base64 string
    question = data["question"]

    payload = {
        "model": VISION_MODEL,
        "prompt": question,
        "images": [image_b64],
        "stream": False
    }
    r = requests.post(f"{OLLAMA_URL}/api/generate", json=payload)
    return {"reply": r.json()["response"]}