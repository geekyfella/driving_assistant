# app.py
import gradio as gr
import joblib
import os
import soundfile as sf
import numpy as np
from transformers import pipeline
import re
from hashlib import md5
from functools import lru_cache
import time

# Globals for lazy loading
ASR_PIPE = None
INTENT_PIPE = None

# Simple in-memory cache {audio_hash: (text, intent, slots)}
TRANS_CACHE = {}

# Config
MAX_SECONDS = 15  # limit audio length to 15s for free-tier safety

INTENT_MODEL_PATH = "intent_clf.pkl"

def load_asr():
    global ASR_PIPE
    if ASR_PIPE is None:
        print("Loading ASR model (this may take 30-60s on first load)...")
        ASR_PIPE = pipeline("automatic-speech-recognition", model="openai/whisper-tiny")
    return ASR_PIPE

def load_intent():
    global INTENT_PIPE
    if INTENT_PIPE is None:
        print("Loading intent classifier...")
        INTENT_PIPE = joblib.load(INTENT_MODEL_PATH)
    return INTENT_PIPE

def audio_duration(filepath):
    data, sr = sf.read(filepath)
    duration = data.shape[0] / float(sr)
    return duration

def get_file_hash(filepath):
    with open(filepath, "rb") as f:
        h = md5(f.read()).hexdigest()
    return h

# Simple slot extractor (destination, contact)
def extract_slots(text):
    slots = {}
    # destination (rudimentary)
    dest_match = re.search(r"\b(home|office|work|downtown|airport|school|college|station|market)\b", text, re.I)
    if dest_match:
        slots['destination'] = dest_match.group(1).lower()
    # contact (names: simple heuristic assume capitalized words or "mom/dad")
    contact_match = re.search(r"\b(mom|dad|mother|father|wife|husband|ravi|raj|anita|michael)\b", text, re.I)
    if contact_match:
        slots['contact'] = contact_match.group(1)
    return slots

def transcribe_and_intent(audio_file):
    # audio_file is a path dict from Gradio (gr.Audio type=filepath)
    if audio_file is None:
        return "No audio", {"intent": "NONE"}, {}

    filepath = audio_file
    # Enforce length
    try:
        dur = audio_duration(filepath)
    except Exception as e:
        return f"Error reading audio: {e}", {"intent": "ERROR"}, {}

    if dur > MAX_SECONDS:
        return f"Audio too long ({dur:.1f}s). Max allowed is {MAX_SECONDS}s.", {"intent": "ERROR_LONG_AUDIO"}, {}

    # cache by file-hash
    h = get_file_hash(filepath)
    if h in TRANS_CACHE:
        transcript, intent, slots = TRANS_CACHE[h]
        source = "cache"
    else:
        asr = load_asr()
        # transformers ASR pipeline accepts path or array depending; pass path
        res = asr(filepath)
        transcript = res.get("text", "").strip()
        intent_model = load_intent()
        intent_label = intent_model.predict([transcript])[0] if transcript else "NONE"
        slots = extract_slots(transcript)
        intent = {"intent": intent_label}
        TRANS_CACHE[h] = (transcript, intent, slots)
        source = "live"

    # Build HTML for TTS button (browser-side)
    escaped = transcript.replace('"', "&quot;").replace("'", "&#39;")
    tts_html = f"""
    <div>
      <button onclick="(function(){{ var t=new SpeechSynthesisUtterance('{escaped}'); window.speechSynthesis.speak(t); }})()">🔊 Play Transcript</button>
      <p style='font-size:12px;color:gray;'>Source: {source}</p>
    </div>
    """
    return transcript, intent["intent"], slots, gr.HTML.update(value=tts_html)

# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("## Vehicle Voice Assistant — Demo (Server-side ASR + Intent)")
    with gr.Row():
        with gr.Column(scale=1):
            audio_in = gr.Audio(source="microphone", type="filepath", label="Record (max 15s)")
            submit = gr.Button("Transcribe & Predict")
            sample_menu = gr.Dropdown(choices=[
                "Take me to my office",
                "Call Mom",
                "Play next song",
                "Stop music",
                "What's the ETA to downtown"
            ], label="Sample utterances")
        with gr.Column(scale=1):
            transcript_out = gr.Textbox(label="Transcript")
            intent_out = gr.Label(label="Predicted Intent")
            slots_out = gr.JSON(label="Extracted Slots")
            tts_html_out = gr.HTML(value="", label="Play (browser TTS)")

    def on_sample(sel):
        # Create a hint for interviewer: they can copy/paste into a recorder or use as test
        return gr.update(value=f"Try saying: {sel}")

    sample_menu.change(fn=on_sample, inputs=sample_menu, outputs=transcript_out)

    def handle_click(audio):
        return transcribe_and_intent(audio)

    submit.click(fn=handle_click, inputs=[audio_in], outputs=[transcript_out, intent_out, slots_out, tts_html_out])

if __name__ == "__main__":
    demo.launch()
