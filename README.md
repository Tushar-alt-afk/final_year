
## 🧠 README.md (Final — Copy This)

```markdown
# 🎙️ Speech-to-Text Benchmarking App (CNN-LSTM vs Gemini)

This project implements a **Speech-to-Text (ASR)** benchmarking web application that compares the performance of two transcription systems:

1. **Local CNN-LSTM / Wav2Vec2 model** (runs locally using TensorFlow or HuggingFace Transformers)
2. **Google Gemini API** (via `google-genai` SDK)

The system transcribes uploaded audio files through both models and displays:
- Word Error Rate (WER)
- Inference Latency
- Cumulative averages
- Dynamic graphs showing performance trends

---

## 📂 Project Structure

```

onelastrif=definfin/
├── backend/
│   ├── app.py                   # Flask backend (main server)
│   ├── requirements.txt         # Python dependencies
│   ├── model/                   # (Optional) CNN-LSTM model files
│   │   ├── cnn_lstm_model.h5
│   │   ├── scaler.pkl
│   │   └── tokenizer.json
│   ├── logs/
│   │   └── records.csv          # Auto-generated logs for all transcriptions
│   └── .env                     # Gemini API Key stored here
│
└── frontend/
└── index.html               # Simple frontend UI

````

---

## ⚙️ Setup Instructions

### 1. Clone this repository
```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>/backend
````

### 2. Create a virtual environment

Windows (PowerShell):

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```

macOS / Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **Tip:** If you get version errors for `google-genai`, use:
>
> ```bash
> pip install -U google-genai flask flask-cors librosa jiwer matplotlib pydub transformers torch tensorflow python-dotenv pandas
> ```

---

## 🔑 Environment Setup

1. In your `backend/` folder, create a `.env` file:

   ```ini
   GEMINI_API_KEY=your_actual_gemini_api_key_here
   ```
2. If you don’t have an API key:

   * Visit [https://aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)
   * Generate one and paste it into `.env`

---

## 🚀 Running the Application

### Step 1: Start the Flask backend

In the `backend/` directory:

```bash
python app.py
```

You’ll see:

```
INFO:asr_app:Starting server on port 8000
 * Running on http://127.0.0.1:8000 (Press CTRL+C to quit)
```

### Step 2: Open the frontend

* Open your browser and visit:
  👉 [http://127.0.0.1:8000](http://127.0.0.1:8000)

### Step 3: Upload and transcribe

1. Upload an audio file (preferably `.wav`, `.mp3`, or `.webm`)
2. Enter a **reference text** (the correct transcript)
   → This is required to calculate Word Error Rate (WER)
3. Click **Transcribe**

✅ The page will display:

* Transcriptions from both systems
* Comparative metrics
* 4 automatically generated charts:

  * WER Trend
  * Inference Time Trend
  * Cumulative Average WER
  * Cumulative Average Time

---

## 📊 Metrics Dashboard (Auto-Generated)

### 1️⃣ WER Trend

Compares the word error rate across multiple audio samples for both models.

### 2️⃣ Inference Time Trend

Shows how long each model took to transcribe each sample.

### 3️⃣ Cumulative Average WER

Displays average accuracy comparison.

### 4️⃣ Cumulative Average Time

Displays average latency over all runs.

Each new transcription automatically updates the CSV logs and regenerates all charts dynamically.

---

## 📁 Logs and Data

All transcription results and metrics are logged automatically in:

```
backend/logs/records.csv
```

Example CSV content:

```csv
timestamp,filename,local_text,gemini_text,local_latency_ms,gemini_latency_ms,reference,local_wer,gemini_wer
1700000000,audio1.wav,hello world,hello word,210.23,98.76,hello world,0.125,0.083
```

---

## 🧩 Example API Usage (Manual cURL Test)

Test without frontend:

```bash
curl -X POST "http://127.0.0.1:8000/transcribe" \
  -F "file=@sample.wav" \
  -F "reference=hello world this is a test"
```

Get only metrics and charts:

```bash
curl http://127.0.0.1:8000/metrics
```

---

## 🧠 Technologies Used

| Layer               | Technologies                                          |
| ------------------- | ----------------------------------------------------- |
| **Backend**         | Flask, Python, TensorFlow, HuggingFace Transformers   |
| **Frontend**        | HTML, JavaScript (Fetch API)                          |
| **Visualization**   | Matplotlib (base64 encoded graphs)                    |
| **Speech Models**   | CNN-LSTM (local), Facebook Wav2Vec2, Gemini 2.5 Flash |
| **Data Processing** | Librosa, JiWER (Word Error Rate), Pandas              |
| **Environment**     | dotenv for API keys                                   |

---

## ⚠️ Common Issues & Fixes

| Issue                               | Cause                               | Fix                                      |
| ----------------------------------- | ----------------------------------- | ---------------------------------------- |
| `No data yet` placeholder in graphs | You didn’t provide a reference text | Always send `reference` in form          |
| `KeyError: latency_ms`              | Gemini returned no valid response   | Check your `.env` API key or quota       |
| `ModuleNotFoundError`               | Dependencies missing                | `pip install -r requirements.txt`        |
| Graphs not appearing on frontend    | Base64 not rendered                 | Ensure frontend uses `img.src = dataURL` |
| Audio decode error                  | Unsupported format                  | Convert to 16kHz mono `.wav`             |

---

## 🧾 Sample `.env` file

```ini
# Backend Configuration
GEMINI_API_KEY=AIzaSyExampleYourKeyHere
```

---

## 💡 Future Enhancements

* Add Whisper API for comparison
* Store results in SQLite or MongoDB
* Real-time streaming transcription
* User authentication and dashboard

---

## 🧑‍💻 Authors

**Team Members**

* Abhimanyu Tiwari (1CR22IS004)
* BS Tushar (1CR22IS035)
* Atiksh Jain (1CR22IS026)

---

## 🪄 License

This project is released under the **MIT License**.
Feel free to use, modify, and distribute it for academic or research purposes.

---

## ⭐ Acknowledgments

Special thanks to:

* [Google AI Studio](https://aistudio.google.com) for Gemini API access
* [Facebook AI Research](https://ai.facebook.com) for Wav2Vec2
* [JiWER](https://github.com/jitsi/jiwer) for evaluation metrics
* VTU ISE Department for guidance

---

**🎯 Developed with Flask + Gemini + Matplotlib by Team ISE**

````

---

## ✅ Optional Additions for GitHub
- Add a top banner image (e.g., `/frontend/banner.png`)  
  ```markdown
  ![Speech-to-Text Benchmarking](frontend/banner.png)
````

* Add badges:

  ```markdown
  ![Python](https://img.shields.io/badge/python-3.10+-blue)
  ![Flask](https://img.shields.io/badge/flask-3.0-lightgrey)
  ![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
  ```

---

