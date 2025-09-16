# app_compat.py  (replace your app.py with this or copy the changes)
import gradio as gr
import joblib
import os
import soundfile as sf
import numpy as np
from transformers import pipeline
import re
from hashlib import md5
import time
import tempfile
import os
import numpy as np
import soundfile as sf

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
    dest_match = re.search(r"\b(home|office|work|downtown|airport|school|college|station|market)\b", text, re.I)
    if dest_match:
        slots['destination'] = dest_match.group(1).lower()
    contact_match = re.search(r"\b(mom|dad|mother|father|wife|husband|ravi|raj|anita|michael)\b", text, re.I)
    if contact_match:
        slots['contact'] = contact_match.group(1)
    return slots

def transcribe_and_intent(audio_file):
    if audio_file is None:
        # return transcript, intent_label, slots, html_string
        return "No audio", "NONE", {}, "<p style='color:gray'>No audio provided</p>"

    filepath = audio_file
    try:
        dur = audio_duration(filepath)
    except Exception as e:
        return f"Error reading audio: {e}", "ERROR", {}, f"<p style='color:red'>Error reading audio: {e}</p>"

    if dur > MAX_SECONDS:
        msg = f"Audio too long ({dur:.1f}s). Max allowed is {MAX_SECONDS}s."
        return msg, "ERROR_LONG_AUDIO", {}, f"<p style='color:red'>{msg}</p>"

    # cache by file-hash
    h = get_file_hash(filepath)
    if h in TRANS_CACHE:
        transcript, intent_label, slots = TRANS_CACHE[h]
        source = "cache"
    else:
        asr = load_asr()
        res = asr(filepath)
        transcript = res.get("text", "").strip()
        intent_model = load_intent()
        intent_label = intent_model.predict([transcript])[0] if transcript else "NONE"
        slots = extract_slots(transcript)
        TRANS_CACHE[h] = (transcript, intent_label, slots)
        source = "live"

    # Build HTML for browser TTS button (safe string)
    escaped = transcript.replace("'", "\\'").replace("\n", " ")
    tts_html = f"""
    <div>
      <button onclick="(function(){{ var t=new SpeechSynthesisUtterance('{escaped}'); window.speechSynthesis.speak(t); }})()">🔊 Play Transcript</button>
      <p style='font-size:12px;color:gray;'>Source: {source}</p>
    </div>
    """
    return transcript, intent_label, slots, tts_html

# Build a Gradio audio component in a backwards-compatible way.
def make_audio_component(label="Record (max 15s)"):
    # Prefer new-style gr.Audio(...) but fallback to gr.inputs.Audio(...) if present or different signature
    try:
        # many modern versions accept source & type
        return gr.Audio(source="microphone", type="filepath", label=label)
    except TypeError:
        # fallback: some versions use different signature or have gr.inputs
        try:
            if hasattr(gr, "inputs") and hasattr(gr.inputs, "Audio"):
                return gr.inputs.Audio(source="microphone", type="filepath", label=label)
        except Exception:
            pass
        # final fallback: simple Audio without source (user may need to upload file)
        try:
            return gr.Audio(label=label)
        except Exception as e:
            raise RuntimeError("Could not create an Audio component for this Gradio version. Consider upgrading gradio with `pip install -U gradio`") from e

# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("## Vehicle Voice Assistant — Demo (Server-side ASR + Intent)")
    with gr.Row():
        with gr.Column(scale=1):
            audio_in = make_audio_component()
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
            intent_out = gr.Textbox(label="Predicted Intent")   # Textbox is more compatible
            slots_out = gr.JSON(label="Extracted Slots")
            tts_html_out = gr.HTML(value="", label="Play (browser TTS)")

    def on_sample(sel):
        return f"Try saying: {sel}"

    sample_menu.change(fn=on_sample, inputs=sample_menu, outputs=transcript_out)

    def handle_click(audio):
        return transcribe_and_intent(audio)

    submit.click(fn=handle_click, inputs=[audio_in], outputs=[transcript_out, intent_out, slots_out, tts_html_out])

# Helper: determine if object is a (sr, np.array) like Gradio returns
def is_audio_tuple(obj):
    return isinstance(obj, (tuple, list)) and len(obj) == 2 and isinstance(obj[0], (int, float)) and isinstance(obj[1], (np.ndarray, list))

def audio_duration(audio_in):
    """
    Accepts:
    - path string -> uses soundfile.read
    - (sr, numpy_array) -> computes duration from len / sr
    - dict with 'name' key (from some gradio versions) -> treat 'name' as path
    """
    # case: tuple (sr, array)
    if is_audio_tuple(audio_in):
        sr, arr = audio_in
        arr = np.asarray(arr)
        return arr.shape[0] / float(sr)
    # case: dict like {"name": "/tmp/xxx.wav", "array": arr, "sampling_rate": sr}
    if isinstance(audio_in, dict):
        # prefer explicit sampling_rate/array
        if "array" in audio_in and "sampling_rate" in audio_in:
            arr = np.asarray(audio_in["array"])
            sr = int(audio_in["sampling_rate"])
            return arr.shape[0] / float(sr)
        # fallback to file path in "name"
        if "name" in audio_in and os.path.exists(audio_in["name"]):
            data, sr = sf.read(audio_in["name"])
            return data.shape[0] / float(sr)
    # case: filepath string
    if isinstance(audio_in, str) and os.path.exists(audio_in):
        data, sr = sf.read(audio_in)
        return data.shape[0] / float(sr)

    raise ValueError(f"Invalid audio input for duration: {type(audio_in)}")

def get_file_hash(audio_in):
    """
    Compute a stable hash for caching:
    - if file path: hash file bytes
    - if (sr, arr): hash arr.bytes + sr
    - if dict: prefer array+sr, else name path
    """
    m = md5()
    if is_audio_tuple(audio_in):
        sr, arr = audio_in
        arr = np.asarray(arr)
        m.update(int(sr).to_bytes(8, "little", signed=False))
        m.update(arr.tobytes())
        return m.hexdigest()
    if isinstance(audio_in, dict):
        if "array" in audio_in and "sampling_rate" in audio_in:
            sr = int(audio_in["sampling_rate"])
            arr = np.asarray(audio_in["array"])
            m.update(int(sr).to_bytes(8, "little", signed=False))
            m.update(arr.tobytes())
            return m.hexdigest()
        if "name" in audio_in and os.path.exists(audio_in["name"]):
            with open(audio_in["name"], "rb") as f:
                m.update(f.read())
            return m.hexdigest()
    if isinstance(audio_in, str) and os.path.exists(audio_in):
        with open(audio_in, "rb") as f:
            m.update(f.read())
        return m.hexdigest()
    # fallback: hash repr
    m.update(repr(audio_in).encode("utf-8"))
    return m.hexdigest()

def _write_temp_wav(sr, arr):
    """Write numpy array to a temp wav file and return path. Caller should remove file."""
    tf = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    tf.close()
    # ensure array is numpy and correct dtype
    arr = np.asarray(arr)
    # soundfile will accept int16 or float32; if floats in -1..1, ok
    sf.write(tf.name, arr, int(sr))
    return tf.name

def transcribe_and_intent(audio_input):
    """
    Robust transcribe_and_intent:
    - Accepts (sr, array), dict, or filepath.
    - Writes temp wav for array inputs, passes path to ASR pipeline.
    - Cleans up temp file after inference.
    """
    if audio_input is None:
        return "No audio", "NONE", {}, "<p style='color:gray'>No audio provided</p>"

    # compute duration (may raise ValueError if invalid)
    try:
        dur = audio_duration(audio_input)
    except Exception as e:
        return f"Error reading audio: {e}", "ERROR", {}, f"<p style='color:red'>Error reading audio: {e}</p>"

    if dur > MAX_SECONDS:
        msg = f"Audio too long ({dur:.1f}s). Max allowed is {MAX_SECONDS}s."
        return msg, "ERROR_LONG_AUDIO", {}, f"<p style='color:red'>{msg}</p>"

    # compute cache hash
    h = get_file_hash(audio_input)
    if h in TRANS_CACHE:
        transcript, intent_label, slots = TRANS_CACHE[h]
        source = "cache"
    else:
        # ensure we have a filepath for ASR pipeline
        tmp_path = None
        try:
            if isinstance(audio_input, str) and os.path.exists(audio_input):
                src_path = audio_input
            elif is_audio_tuple(audio_input):
                sr, arr = audio_input
                tmp_path = _write_temp_wav(sr, arr)
                src_path = tmp_path
            elif isinstance(audio_input, dict):
                # prefer explicit array + sampling_rate
                if "array" in audio_input and "sampling_rate" in audio_input:
                    sr = int(audio_input["sampling_rate"])
                    arr = np.asarray(audio_input["array"])
                    tmp_path = _write_temp_wav(sr, arr)
                    src_path = tmp_path
                elif "name" in audio_input and os.path.exists(audio_input["name"]):
                    src_path = audio_input["name"]
                else:
                    raise ValueError("Unsupported dict audio format")
            else:
                raise ValueError("Unsupported audio input type")

            asr = load_asr()
            # pass filepath to ASR pipeline
            res = asr(src_path)
            transcript = res.get("text", "").strip()
            intent_model = load_intent()
            intent_label = intent_model.predict([transcript])[0] if transcript else "NONE"
            slots = extract_slots(transcript)
            TRANS_CACHE[h] = (transcript, intent_label, slots)
            source = "live"
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass

    # Build TTS html
    escaped = (TRANS_CACHE[h][0] if h in TRANS_CACHE else "").replace("'", "\\'").replace("\n", " ")
    tts_html = f"""
    <div>
      <button onclick="(function(){{ var t=new SpeechSynthesisUtterance('{escaped}'); window.speechSynthesis.speak(t); }})()">🔊 Play Transcript</button>
      <p style='font-size:12px;color:gray;'>Source: {source}</p>
    </div>
    """
    transcript, intent_label, slots = TRANS_CACHE[h]
    return transcript, intent_label, slots, tts_html

if __name__ == "__main__":
    demo.launch()
