import os
import time
import tempfile
import traceback
import json
from pathlib import Path
from flask import Flask, request, jsonify
from pydub import AudioSegment
import librosa
import numpy as np
from jiwer import wer as jiwer_wer
from flask_cors import CORS
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import csv
import pandas as pd
from dotenv import load_dotenv
import logging
import sys
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger("asr_app")

# try multiple genai import names
genai = None
try:
    from google import genai as _g
    genai = _g
except Exception:
    try:
        import google.generativeai as _g
        genai = _g
    except Exception:
        try:
            import google.genai as _g
            genai = _g
        except Exception:
            genai = None

BASE = Path(__file__).resolve().parent
LOGS_DIR = BASE / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)
RECORDS_CSV = LOGS_DIR / "records.csv"

# --- CSV file fix/reset helper ---
def fix_or_create_csv(path):
    expected_header = ["timestamp","filename","local_text","gemini_text",
                       "local_latency_ms","gemini_latency_ms","reference",
                       "local_wer","gemini_wer"]
    try:
        with open(path, "r", newline='', encoding="utf-8") as f:
            reader = csv.reader(f)
            rows = list(reader)
        # If CSV is empty or header incorrect, recreate
        if not rows or rows[0] != expected_header:
            print(f"CSV header invalid or missing. Recreating {path}.")
            with open(path, "w", newline='', encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(expected_header)
    except Exception as e:
        print(f"Error reading CSV {path}: {e}. Creating fresh file.")
        with open(path, "w", newline='', encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(expected_header)

# Fix or create CSV file on startup to ensure proper header
fix_or_create_csv(RECORDS_CSV)

MODEL_DIR = BASE / "model"
MODEL_PATH = MODEL_DIR / "cnn_lstm_model.h5"
SCALER_PATH = MODEL_DIR / "scaler.pkl"
TOKENIZER_PATH = MODEL_DIR / "tokenizer.json"

app = Flask(__name__, static_folder=str(BASE.parent / "frontend"), static_url_path="/")
CORS(app, supports_credentials=True)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024

TF_AVAILABLE = False
TF_MODEL = None
SCALER = None
TOKENIZER = None
WAV2VEC_LOADED = False

logger.info("GENAI SDK present: %s", genai is not None)
logger.info("GEMINI_API_KEY present: %s", bool(os.getenv("GEMINI_API_KEY")))

# Try to load TF model components (optional)
try:
    import tensorflow as tf
    if MODEL_PATH.exists() and MODEL_PATH.stat().st_size > 0:
        TF_MODEL = tf.keras.models.load_model(str(MODEL_PATH))
    if SCALER_PATH.exists() and SCALER_PATH.stat().st_size > 0:
        with open(SCALER_PATH, "rb") as f:
            SCALER = pickle.load(f)
    if TOKENIZER_PATH.exists() and TOKENIZER_PATH.stat().st_size > 0:
        with open(TOKENIZER_PATH, "r", encoding="utf-8") as f:
            TOKENIZER = json.load(f)
    if TF_MODEL is not None and SCALER is not None and TOKENIZER is not None:
        TF_AVAILABLE = True
        logger.info("Local TF CNN-LSTM model loaded and available.")
    else:
        logger.info("Local TF model incomplete or missing; will fallback to Wav2Vec2.")
except Exception as e:
    logger.info("TensorFlow not loaded or model missing: %s", e)
    TF_AVAILABLE = False

# ----------------- Utilities -----------------
def save_bytes_to_tempfile(b, suffix=".webm"):
    tf = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tf.write(b)
    tf.flush()
    tf.close()
    return tf.name

def convert_to_standard_wav(in_path):
    out = tempfile.mktemp(suffix=".wav")
    audio = AudioSegment.from_file(in_path)
    # Ensure 16kHz for standard ASR models
    audio.export(out, format="wav", parameters=["-ar", "16000"]) 
    return out

def preprocess_audio_for_inference(wav_path, scaler):
    try:
        y, sr = librosa.load(wav_path, sr=16000, mono=True)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        mfcc_t = mfcc.T
        max_len = 300
        if mfcc_t.shape[0] >= max_len:
            mfcc_t = mfcc_t[:max_len, :]
        else:
            pad = np.zeros((max_len - mfcc_t.shape[0], mfcc_t.shape[1]), dtype=np.float32)
            mfcc_t = np.vstack([mfcc_t, pad])
        arr = mfcc_t.astype('float32')[None, :, :]
        if scaler is not None:
            s = arr.reshape(-1, arr.shape[-1])
            s2 = scaler.transform(s)
            arr = s2.reshape(arr.shape)
        return arr
    except Exception as e:
        logger.error("preprocess error: %s", e)
        return None

def tf_ctc_greedy_decode(preds, tokenizer_json):
    try:
        import numpy as _np
        if preds is None:
            return ""
        if preds.ndim == 3:
            seq = _np.argmax(preds, axis=-1)[0]
        elif preds.ndim == 2:
            seq = _np.argmax(preds, axis=-1)
        else:
            return ""
        index_word = {}
        data = tokenizer_json
        if isinstance(data, dict):
            if "index_word" in data and isinstance(data["index_word"], dict):
                index_word = {int(k): v for k, v in data["index_word"].items()}
            elif "word_index" in data and isinstance(data["word_index"], dict):
                for k, v in data["word_index"].items():
                    index_word[int(v)] = k
        out_chars = []
        prev = None
        for idx in seq:
            idx = int(idx)
            if idx == 0:
                prev = idx
                continue
            if idx == prev:
                continue
            ch = index_word.get(idx, "")
            out_chars.append(ch)
            prev = idx
        text = "".join(out_chars).strip()
        return " ".join(text.split())
    except Exception as e:
        logger.error("tf_ctc_greedy_decode error: %s", e)
        return ""

# ----------------- ASR FUNCTION -----------------
def run_local_inference(wav_path):
    global TF_AVAILABLE, TF_MODEL, SCALER, TOKENIZER, WAV2VEC_LOADED
    
    # 1. Try Local TF CNN-LSTM Model
    if TF_AVAILABLE and TF_MODEL is not None:
        try:
            arr = preprocess_audio_for_inference(wav_path, SCALER)
            if arr is None:
                raise RuntimeError("preprocess failed")
            start = time.time()
            # Use silent prediction to prevent stdout spam
            preds = TF_MODEL.predict(arr, verbose=0) 
            latency = (time.time() - start) * 1000.0
            transcription = tf_ctc_greedy_decode(preds, TOKENIZER)
            if transcription and transcription.strip():
                return {"text": transcription, "latency_ms": round(latency, 2), "provider": "cnn-lstm", "error": None}
        except Exception as e:
            logger.warning("TF model inference failed or empty: %s", e)
            TF_AVAILABLE = False # Disable TF model if it fails once

    # 2. Fallback to Wav2Vec2 HuggingFace Model
    try:
        if not WAV2VEC_LOADED:
            # Note: This loads the model/processor only once
            run_local_inference._proc = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
            run_local_inference._model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
            WAV2VEC_LOADED = True
            
        proc = run_local_inference._proc
        model = run_local_inference._model
        
        # Load audio data to 16kHz for Wav2Vec2
        y, sr = librosa.load(wav_path, sr=16000, mono=True)
        input_values = proc(y, sampling_rate=16000, return_tensors="pt").input_values
        
        start = time.time()
        with torch.no_grad():
            logits = model(input_values).logits
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = proc.batch_decode(predicted_ids)[0].lower()
        latency = (time.time() - start) * 1000.0
        
        return {"text": transcription, "latency_ms": round(latency, 2), "provider": "wav2vec2", "error": None}
        
    except Exception as e:
        logger.error("Wav2Vec2 inference error: %s", e)
        return {"text": None, "latency_ms": None, "provider": "none", "error": str(e)}

# ----------------- GEMINI FUNCTION (FIXED SERIALIZATION) -----------------
def call_gemini_transcribe(gemini_api_key, filepath, timeout=60):
    if not gemini_api_key:
        return {"text": None, "latency_ms": None, "provider": "none", "error": "no key", "raw": None}
    if not genai:
        return {"text": None, "latency_ms": None, "provider": "none", "error": "genai sdk missing", "raw": None}
    start = time.time()
    client = None
    try:
        # Standard client initialization logic with fallbacks
        if hasattr(genai, "Client"):
            client = genai.Client(api_key=gemini_api_key)
        elif hasattr(genai, "client") and hasattr(genai.client, "Client"):
            client = genai.client.Client(api_key=gemini_api_key)
        else:
            try:
                import importlib
                g2 = importlib.import_module("google.genai")
                client = g2.Client(api_key=gemini_api_key) if hasattr(g2, "Client") else None
            except Exception:
                client = None
        if client is None:
            return {"text": None, "latency_ms": None, "provider": "gemini-sdk-failed", "error": "client constructor not found", "raw": None}
    except Exception as e:
        return {"text": None, "latency_ms": None, "provider": "gemini-sdk-failed", "error": f"client init err: {e}", "raw": None}

    audio_file = None
    try:
        audio_file = client.files.upload(file=filepath)
        models_to_try = ['gemini-2.5-flash']
        
        for model_name in models_to_try:
            try:
                # Actual Gemini API call for transcription
                response = client.models.generate_content(model=model_name, contents=[audio_file, "Please transcribe the audio exactly, only text."])
                
                text = None
                # Robust response parsing
                try:
                    text = response.text.strip()
                except Exception:
                    text = getattr(response, "text", None)
                
                if text:
                    latency = (time.time() - start) * 1000.0
                    try:
                        client.files.delete(name=audio_file.name)
                    except Exception: pass
                    
                    # *** CRITICAL FIX: Convert the non-serializable object to a string ***
                    raw_content = str(response) 
                    
                    return {"text": text, "latency_ms": round(latency, 2), "provider": model_name, "error": None, "raw": raw_content} # 'raw' is now a string
            except Exception:
                continue
                
        latency = (time.time() - start) * 1000.0
        try:
            client.files.delete(name=audio_file.name)
        except Exception: pass
        return {"text": None, "latency_ms": round(latency, 2), "provider": "gemini-none", "error": "no transcription returned", "raw": None}
        
    except Exception as e:
        latency = (time.time() - start) * 1000.0
        try:
            if audio_file and hasattr(client, "files"):
                client.files.delete(name=audio_file.name)
        except Exception: pass
        return {"text": None, "latency_ms": round(latency, 2), "provider": "gemini-sdk-failed", "error": str(e), "raw": None}

# ----------------- CSV helpers -----------------
def create_or_append_record(row: dict):
    header = ["timestamp","filename","local_text","gemini_text","local_latency_ms","gemini_latency_ms","reference","local_wer","gemini_wer"]
    new_file = not RECORDS_CSV.exists()
    with open(RECORDS_CSV, "a", newline='', encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if new_file:
            writer.writeheader()
        writer.writerow({k: ("" if row.get(k) is None else row.get(k)) for k in header})

def read_records_df():
    cols = ["timestamp","filename","local_text","gemini_text","local_latency_ms","gemini_latency_ms","reference","local_wer","gemini_wer"]
    if not RECORDS_CSV.exists():
        print(f"{RECORDS_CSV} not found, returning empty DataFrame")
        return pd.DataFrame(columns=cols)
    try:
        df = pd.read_csv(RECORDS_CSV)
        for c in cols:
            if c not in df.columns:
                df[c] = np.nan
        for c in ["local_latency_ms","gemini_latency_ms","local_wer","gemini_wer"]:
            df[c] = pd.to_numeric(df[c], errors='coerce')

        print(f"Read DataFrame with {len(df)} rows")
        print(df.tail())
        
        return df[cols]
    except Exception as e:
        logger.error("read_records_df error: %s", e)
        return pd.DataFrame(columns=cols)

# --- STATS HELPER (CRITICAL FIX) ---
def safe_mean(series):
    """Safely calculates mean, returning None if the series is empty, for robust JSON serialization."""
    if series.dropna().empty:
        return None
    return float(series.mean()) 

# ----------------- Chart functions (FINAL FIXED) -----------------
def _placeholder_image(text="No data available"):
    fig, ax = plt.subplots(figsize=(6,2))
    ax.text(0.5, 0.5, text, ha='center', va='center', fontsize=12)
    ax.axis('off')
    buf = BytesIO(); fig.savefig(buf, format='png', bbox_inches='tight'); plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode('utf-8')

def generate_line_chart_wers(df):
    """1. WER Trend: Ensures visibility for a single data point."""
    if df.empty or df[['local_wer','gemini_wer']].dropna(how='all').empty:
        return _placeholder_image("No WER data yet. Provide reference text to compute WER.")
        
    fig, ax = plt.subplots(figsize=(10,4))
    x = list(range(1, len(df)+1))
    
    x_ticks = x if len(x) > 1 else [1]
    
    ax.plot(x, df['local_wer'].fillna(np.nan).tolist(), marker='o', label='Local WER')
    ax.plot(x, df['gemini_wer'].fillna(np.nan).tolist(), marker='o', label='Gemini WER')
    
    ax.set_xlabel('Sample #'); ax.set_ylabel('WER'); ax.set_title('1. WER Trend (Local vs Gemini)')
    ax.legend()
    
    ax.set_xticks(x_ticks)
    ax.set_xlim(0.5, max(1.5, len(df) + 0.5)) 
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, linestyle='--', alpha=0.6)
    
    buf = BytesIO(); fig.savefig(buf, format='png', bbox_inches='tight'); plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode('utf-8')

def generate_bar_chart_times(df):
    """2. Time Trend: Ensures dynamic Y-axis and proper X-axis for a single sample."""
    if df.empty or df[['local_latency_ms','gemini_latency_ms']].dropna(how='all').empty:
        return _placeholder_image("No latency data yet.")
        
    fig, ax = plt.subplots(figsize=(10,4))
    indices = np.arange(len(df))
    width = 0.35
    local_times = df['local_latency_ms'].fillna(0).tolist()
    gem_times = df['gemini_latency_ms'].fillna(0).tolist()
    
    ax.bar(indices - width/2, local_times, width, label='Local (ms)', color='#2563eb')
    ax.bar(indices + width/2, gem_times, width, label='Gemini (ms)', color='#06b6d4')
    
    ax.set_xlabel('Sample #'); ax.set_ylabel('Latency (ms)'); ax.set_title('2. Inference Time Trend')
    ax.set_xticks(indices)
    ax.legend()
    
    # Set X-limits to center the bar(s)
    ax.set_xlim(-width, len(df) - 1 + width * 2)
    
    # Dynamically set Y-limit based on the largest recorded time, with a minimum floor
    max_time = df[['local_latency_ms','gemini_latency_ms']].max().max()
    y_limit = max(1000, max_time * 1.1)
    ax.set_ylim(0, y_limit) 
    
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    buf = BytesIO(); fig.savefig(buf, format='png', bbox_inches='tight'); plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode('utf-8')

def generate_cumulative_avg_bar(df, col_local, col_gem, title, ylabel):
    """3 & 4. Cumulative Avg: Ensures visible Y-axis range and visible text annotation."""
    if df.empty:
        return _placeholder_image("No data to compute averages.")
        
    avg_local = float(df[col_local].dropna().mean()) if not df[col_local].dropna().empty else 0.0
    avg_gem = float(df[col_gem].dropna().mean()) if not df[col_gem].dropna().empty else 0.0
    
    bar_title = '3. Cumulative Average WER' if 'wer' in col_local else '4. Cumulative Average Time (ms)'
    is_time = 'latency' in title.lower() or 'time' in title.lower()
    
    fig, ax = plt.subplots(figsize=(6,4))
    bars = ax.bar(['Local','Gemini'], [avg_local, avg_gem], color=['#2563eb','#06b6d4'])
    
    if is_time:
        # Dynamic Y-limit for time with a minimum floor of 1000ms
        max_avg = max(avg_local, avg_gem, 1000) 
        ax.set_ylim(0, max_avg * 1.1)
    else:
        # Dynamic Y-limit for WER with a minimum ceiling of 1.05
        max_avg = max(avg_local, avg_gem)
        ax.set_ylim(0, max(1.05, max_avg * 1.1))

    for b in bars:
        h = b.get_height()
        # Ensure annotation is visible even on small bars
        annotation_y = h + max(ax.get_ylim()[1] * 0.01, h * 0.05) # Padding is 1% of axis height or 5% of bar height
        
        ax.text(b.get_x()+b.get_width()/2, 
                annotation_y, 
                f"{h:.3f}" if not is_time else f"{h:.2f}", 
                ha='center', va='bottom', fontsize=9)
                
    ax.set_ylabel(ylabel); ax.set_title(bar_title); ax.grid(axis='y', linestyle='--', alpha=0.6)
    buf = BytesIO(); fig.savefig(buf, format='png', bbox_inches='tight'); plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode('utf-8')

# ----------------- Endpoints -----------------
@app.route('/transcribe', methods=['POST'])
def transcribe_endpoint():
    tmp_file_path = None
    wav_path = None
    try:
        gemkey = os.getenv("GEMINI_API_KEY")
        reference = request.form.get("reference")
        filename = None

        if 'file' in request.files:
            f = request.files['file']
            filename = f.filename
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=Path(f.filename).suffix)
            f.save(tmp.name); tmp.flush(); tmp.close()
            tmp_file_path = tmp.name
            wav_path = convert_to_standard_wav(tmp_file_path)
        elif request.data and len(request.data) > 0:
            tmp_file_path = save_bytes_to_tempfile(request.data, suffix=".webm")
            wav_path = convert_to_standard_wav(tmp_file_path)
            filename = Path(wav_path).name
        else:
            return jsonify({"error": "No audio provided"}), 400

        local_res = run_local_inference(wav_path) or {}
        results = {'local': local_res}
        
        if gemkey:
            gem_res = call_gemini_transcribe(gemkey, wav_path) or {}
            results['gemini'] = gem_res
        else:
            results['gemini'] = {"text": None, "latency_ms": None, "provider": "none", "error": "no gemini key", "raw": None}


        local_text = (local_res.get('text') or "").strip()
        gem_text = (results['gemini'].get('text') or "").strip()
        reference = (reference or "").strip()

        local_wer = None
        gem_wer = None
        if reference:
            try:
                local_wer = jiwer_wer(reference.lower(), local_text.lower()) if local_text else None
            except Exception: pass
            try:
                gem_wer = jiwer_wer(reference.lower(), gem_text.lower()) if gem_text else None
            except Exception: pass

        row = {
            "timestamp": time.time(), "filename": filename or "", "local_text": local_text, 
            "gemini_text": gem_text, "local_latency_ms": local_res.get('latency_ms', None),
            "gemini_latency_ms": results['gemini'].get('latency_ms', None), "reference": reference,
            "local_wer": local_wer, "gemini_wer": gem_wer
        }
        create_or_append_record(row)

        df = read_records_df()
        metrics = {
            "count": len(df), 
            "last": row,
            "charts": {
                "wer_trend": generate_line_chart_wers(df),
                "time_trend": generate_bar_chart_times(df),
                "avg_wer": generate_cumulative_avg_bar(df, 'local_wer', 'gemini_wer', 'Cumulative Average WER', 'WER'),
                "avg_time": generate_cumulative_avg_bar(df, 'local_latency_ms', 'gemini_latency_ms', 'Cumulative Average Time (ms)', 'Latency (ms)')
            },
            "stats": {
                "avg_local_wer": safe_mean(df['local_wer']),
                "avg_gemini_wer": safe_mean(df['gemini_wer']),
                "avg_local_time_ms": safe_mean(df['local_latency_ms']),
                "avg_gemini_time_ms": safe_mean(df['gemini_latency_ms'])
            }
        }

        results['metrics'] = metrics
        results['meta'] = {"audio_path": filename or ""}

        return jsonify(results)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
    finally:
        try:
            if tmp_file_path and os.path.exists(tmp_file_path):
                os.remove(tmp_file_path)
        except Exception:
            pass
        try:
            if wav_path and os.path.exists(wav_path) and wav_path != tmp_file_path:
                os.remove(wav_path)
        except Exception:
            pass

@app.route('/metrics', methods=['GET'])
def metrics_endpoint():
    df = read_records_df()
    response = {
        "count": len(df),
        "charts": {
            "wer_trend": generate_line_chart_wers(df),
            "time_trend": generate_bar_chart_times(df),
            "avg_wer": generate_cumulative_avg_bar(df, 'local_wer', 'gemini_wer', 'Cumulative Average WER', 'WER'),
            "avg_time": generate_cumulative_avg_bar(df, 'local_latency_ms', 'gemini_latency_ms', 'Cumulative Average Time (ms)', 'Latency (ms)')
        },
        "stats": {
            "avg_local_wer": safe_mean(df['local_wer']),
            "avg_gemini_wer": safe_mean(df['gemini_wer']),
            "avg_local_time_ms": safe_mean(df['local_latency_ms']),
            "avg_gemini_time_ms": safe_mean(df['gemini_latency_ms'])
        }
    }
    return jsonify(response)

@app.route('/', methods=['GET'])
def index():
    return app.send_static_file('index.html')

if __name__ == '__main__':
    logger.info("Starting server on port 8000")
    app.run(debug=True, host='0.0.0.0', port=8000)
