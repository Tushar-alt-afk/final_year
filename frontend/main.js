const recordButton = document.getElementById('recordButton');
const stopButton = document.getElementById('stopButton');
const statusMessage = document.getElementById('statusMessage');
const referenceText = document.getElementById('referenceText');

const localRow = document.getElementById('localRow');
const geminiRow = document.getElementById('geminiRow');

const sampleCount = document.getElementById('sampleCount');
const avgLocalWer = document.getElementById('avgLocalWer');
const avgGeminiWer = document.getElementById('avgGeminiWer');
const avgLocalTime = document.getElementById('avgLocalTime');
const avgGeminiTime = document.getElementById('avgGeminiTime');

const chartWerTrend = document.getElementById('chartWerTrend');
const chartTimeTrend = document.getElementById('chartTimeTrend');
const chartAvgWer = document.getElementById('chartAvgWer');
const chartAvgTime = document.getElementById('chartAvgTime');

let mediaRecorder;
let audioChunks = [];
const apiUrl = 'http://127.0.0.1:8000';

function updateRow(row, data) {
    const cells = row.querySelectorAll('td');
    cells[0].textContent = data.provider || (data.error ? 'Error' : '-');
    cells[1].textContent = data.text || data.error || 'No transcription';
    cells[2].textContent = data.latency_ms !== null ? data.latency_ms.toFixed(2) : '-';
    cells[3].textContent = data.wer !== null ? data.wer.toFixed(4) : '-';
}

function updateMetrics(metrics) {
    const stats = metrics.stats;
    const charts = metrics.charts;

    sampleCount.textContent = metrics.count;
    avgLocalWer.textContent = stats.avg_local_wer !== null ? stats.avg_local_wer.toFixed(4) : '-';
    avgGeminiWer.textContent = stats.avg_gemini_wer !== null ? stats.avg_gemini_wer.toFixed(4) : '-';
    avgLocalTime.textContent = stats.avg_local_time_ms !== null ? stats.avg_local_time_ms.toFixed(2) : '-';
    avgGeminiTime.textContent = stats.avg_gemini_time_ms !== null ? stats.avg_gemini_time_ms.toFixed(2) : '-';

    chartWerTrend.src = charts.wer_trend || '';
    chartTimeTrend.src = charts.time_trend || '';
    chartAvgWer.src = charts.avg_wer || '';
    chartAvgTime.src = charts.avg_time || '';
}

async function fetchMetrics() {
    try {
        const response = await fetch(`${apiUrl}/metrics`);
        const data = await response.json();
        updateMetrics(data);
    } catch (error) {
        console.error('Error fetching metrics:', error);
    }
}

// Initial fetch of metrics
fetchMetrics();

recordButton.onclick = async () => {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaRecorder = new MediaRecorder(stream);
        audioChunks = [];

        mediaRecorder.ondataavailable = event => {
            audioChunks.push(event.data);
        };

        mediaRecorder.onstop = () => {
            const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
            transcribeAudio(audioBlob);
            stream.getTracks().forEach(track => track.stop());
        };

        mediaRecorder.start();
        statusMessage.textContent = 'Recording... Say something!';
        recordButton.disabled = true;
        stopButton.disabled = false;
    } catch (err) {
        statusMessage.textContent = `Error accessing microphone: ${err}`;
        console.error('Microphone error:', err);
    }
};

stopButton.onclick = () => {
    mediaRecorder.stop();
    statusMessage.textContent = 'Recording stopped. Transcribing...';
    stopButton.disabled = true;
};

async function transcribeAudio(audioBlob) {
    recordButton.disabled = true;
    stopButton.disabled = true;
    statusMessage.textContent = 'Transcribing... please wait.';

    const formData = new FormData();
    formData.append('file', audioBlob, 'audio.webm');
    formData.append('reference', referenceText.value);

    try {
        const response = await fetch(`${apiUrl}/transcribe`, {
            method: 'POST',
            body: formData,
        });

        const data = await response.json();
        
        // Handle the case where Gemini returns a definitive error
        const geminiError = data.gemini.error;
        if (geminiError && geminiError.includes('API Key not found')) {
            alert("API Key Error: Please check your backend console. The Gemini API key is missing or invalid. Did you set it in .env?");
            statusMessage.textContent = 'Transcription failed due to API Key error. Check console.';
        } else {
            statusMessage.textContent = 'Transcription complete!';
        }

        updateRow(localRow, {
            provider: data.local.provider,
            text: data.local.text,
            latency_ms: data.local.latency_ms,
            wer: data.metrics.last.local_wer
        });

        updateRow(geminiRow, {
            provider: data.gemini.provider,
            text: data.gemini.text,
            latency_ms: data.gemini.latency_ms,
            wer: data.metrics.last.gemini_wer
        });

        updateMetrics(data.metrics);

    } catch (error) {
        statusMessage.textContent = `A network error occurred: ${error}`;
        console.error('Fetch error:', error);
    } finally {
        recordButton.disabled = false;
    }
}