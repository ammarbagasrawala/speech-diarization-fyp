# app.py
import os
import uuid
import time
import shutil
import numpy as np
import torch
import librosa
import soundfile as sf
from pyannote.audio import Pipeline
from pydub import AudioSegment
import whisper
import warnings
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException, Request, Form
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import Dict, List, Optional
import uvicorn
import logging

DEFAULT_HF_TOKEN = os.environ.get("HF_TOKEN", "")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore")

# Create FastAPI instance
app = FastAPI(
    title="Speaker Diarization and Transcription API",
    description="API for audio speaker diarization and transcription",
    version="1.0.0"
)

# Create necessary directories
UPLOAD_DIR = os.path.join(os.getcwd(), "uploads")
RESULTS_DIR = os.path.join(os.getcwd(), "results")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Create templates directory for basic UI
templates_dir = os.path.join(os.getcwd(), "templates")
os.makedirs(templates_dir, exist_ok=True)

# Create templates
templates = Jinja2Templates(directory=templates_dir)

# Create a simple HTML file for the UI
with open(os.path.join(templates_dir, "index.html"), "w") as f:
    f.write("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Speaker Diarization and Transcription</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
            .container { border: 1px solid #ddd; padding: 20px; border-radius: 5px; }
            .form-group { margin-bottom: 15px; }
            label { display: block; margin-bottom: 5px; }
            input, button { padding: 8px; }
            button { background-color: #4CAF50; color: white; border: none; cursor: pointer; }
            #status { margin-top: 20px; padding: 10px; display: none; }
            .success { background-color: #d4edda; border-color: #c3e6cb; color: #155724; }
            .error { background-color: #f8d7da; border-color: #f5c6cb; color: #721c24; }
            #results { margin-top: 20px; }
            pre { background-color: #f8f9fa; padding: 10px; border-radius: 5px; overflow-x: auto; }
        </style>
    </head>
    <body>
        <h1>Speaker Diarization and Transcription</h1>
        <div class="container">
            <form id="uploadForm" enctype="multipart/form-data">
                <div class="form-group">
                    <label for="audioFile">Upload Audio File (WAV, MP3, etc.):</label>
                    <input type="file" id="audioFile" name="audioFile" accept="audio/*" required>
                </div>
                <div class="form-group">
                    <input type="hidden" id="hfToken" name="hfToken" value="{DEFAULT_HF_TOKEN}">
                </div>
                <button type="submit">Process Audio</button>
            </form>
            
            <div id="status"></div>
            
            <div id="results">
                <h2>Results</h2>
                <div id="jobStatus"></div>
                <div id="downloadLinks"></div>
                <div id="transcriptions"></div>
            </div>
        </div>

        <script>
            document.getElementById('uploadForm').addEventListener('submit', async function(e) {
                e.preventDefault();
                
                const formData = new FormData();
                const audioFile = document.getElementById('audioFile').files[0];
                const hfToken = document.getElementById('hfToken').value;
                
                formData.append('audio_file', audioFile);
                if (hfToken) {
                    formData.append('hf_token', hfToken);
                }
                
                const status = document.getElementById('status');
                status.style.display = 'block';
                status.className = '';
                status.textContent = 'Processing... This may take a few minutes.';
                
                try {
                    const response = await fetch('/process/', {
                        method: 'POST',
                        body: formData
                    });
                    
                    const result = await response.json();
                    
                    if (result.job_id) {
                        status.className = 'success';
                        status.textContent = 'Job submitted successfully! Checking status...';
                        checkJobStatus(result.job_id);
                    } else {
                        status.className = 'error';
                        status.textContent = 'Error: ' + result.detail;
                    }
                } catch (error) {
                    status.className = 'error';
                    status.textContent = 'Error: ' + error.message;
                }
            });
            
            async function checkJobStatus(jobId) {
                const jobStatus = document.getElementById('jobStatus');
                
                while (true) {
                    try {
                        const response = await fetch(`/status/${jobId}`);
                        const result = await response.json();
                        
                        jobStatus.textContent = `Status: ${result.status}`;
                        
                        if (result.status === 'completed') {
                            document.getElementById('status').textContent = 'Processing completed successfully!';
                            
                            // Display download links
                            const downloadLinks = document.getElementById('downloadLinks');
                            downloadLinks.innerHTML = '<h3>Download Files</h3><ul>';
                            
                            for (const [speaker, path] of Object.entries(result.speaker_audio_paths || {})) {
                                downloadLinks.innerHTML += `<li><a href="/download/${jobId}/${speaker}" download>${speaker}.wav</a></li>`;
                            }
                            
                            downloadLinks.innerHTML += `<li><a href="/download/${jobId}/transcriptions.txt" download>transcriptions.txt</a></li>`;
                            downloadLinks.innerHTML += '</ul>';
                            
                            // Display transcriptions
                            const transcriptions = document.getElementById('transcriptions');
                            transcriptions.innerHTML = '<h3>Transcriptions</h3><pre>';
                            
                            for (const [speaker, text] of Object.entries(result.transcriptions || {})) {
                                transcriptions.innerHTML += `${speaker}: ${text}\n\n`;
                            }
                            
                            transcriptions.innerHTML += '</pre>';
                            
                            break;
                        } else if (result.status === 'failed') {
                            document.getElementById('status').className = 'error';
                            document.getElementById('status').textContent = 'Processing failed: ' + result.error;
                            break;
                        }
                        
                        // Wait for 3 seconds before checking again
                        await new Promise(resolve => setTimeout(resolve, 3000));
                    } catch (error) {
                        jobStatus.textContent = `Error checking status: ${error.message}`;
                        break;
                    }
                }
            }
        </script>
    </body>
    </html>
    """)

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Store job information
jobs = {}

class Job:
    def __init__(self, job_id, audio_path, hf_token=None):
        self.job_id = job_id
        self.audio_path = audio_path
        self.hf_token = hf_token
        self.status = "pending"
        self.result_dir = os.path.join(RESULTS_DIR, job_id)
        self.speaker_audio_paths = {}
        self.transcriptions = {}
        self.error = None
        os.makedirs(self.result_dir, exist_ok=True)

    def to_dict(self):
        return {
            "job_id": self.job_id,
            "status": self.status,
            "speaker_audio_paths": {k: os.path.basename(v) for k, v in self.speaker_audio_paths.items()} if self.speaker_audio_paths else {},
            "transcriptions": self.transcriptions,
            "error": self.error
        }


def convert_to_wav(audio_path, output_path=None):
    """Convert audio to WAV format if needed"""
    if audio_path.endswith('.wav'):
        return audio_path
    
    if output_path is None:
        output_path = os.path.splitext(audio_path)[0] + '.wav'
    
    audio = AudioSegment.from_file(audio_path)
    audio.export(output_path, format='wav')
    logger.info(f"Converted {audio_path} to WAV format: {output_path}")
    return output_path


def perform_diarization(audio_path, hf_token=None):
    """Perform speaker diarization on audio file"""
    logger.info("Performing speaker diarization...")
    
    if hf_token is None:
        logger.warning("No HuggingFace token provided. You may face rate limits.")
    
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.0",
        use_auth_token=hf_token
    )
    
    diarization = pipeline(audio_path)
    
    speaker_segments = {}
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        if speaker not in speaker_segments:
            speaker_segments[speaker] = []
        speaker_segments[speaker].append((turn.start, turn.end))
    
    for speaker in speaker_segments:
        merged_segments = []
        sorted_segments = sorted(speaker_segments[speaker])
        for segment in sorted_segments:
            if not merged_segments or segment[0] > merged_segments[-1][1]:
                merged_segments.append(segment)
            else:
                merged_segments[-1] = (merged_segments[-1][0], max(merged_segments[-1][1], segment[1]))
        speaker_segments[speaker] = merged_segments
    
    logger.info(f"Found {len(speaker_segments)} speakers in the audio.")
    return speaker_segments


def extract_speaker_audio(audio_path, speaker_segments, output_dir):
    """Extract audio for each speaker"""
    logger.info("Extracting audio for each speaker...")
    audio, sr = librosa.load(audio_path, sr=None, mono=True)
    os.makedirs(output_dir, exist_ok=True)
    
    speaker_audio_paths = {}
    
    for speaker, segments in speaker_segments.items():
        speaker_audio = np.zeros_like(audio)
        for start, end in segments:
            start_sample = int(start * sr)
            end_sample = int(end * sr)
            if end_sample > len(audio):
                end_sample = len(audio)
            if start_sample >= end_sample:
                continue
            speaker_audio[start_sample:end_sample] = audio[start_sample:end_sample]
        
        if np.max(np.abs(speaker_audio)) > 0:
            speaker_audio = speaker_audio / np.max(np.abs(speaker_audio)) * 0.9
        
        output_path = os.path.join(output_dir, f"{speaker.replace('SPEAKER_', 'speaker_')}.wav")
        sf.write(output_path, speaker_audio, sr)
        speaker_audio_paths[speaker] = output_path
        logger.info(f"Saved {speaker} audio to {output_path}")
    
    return speaker_audio_paths


def transcribe_audio(audio_paths):
    """Transcribe audio for each speaker"""
    logger.info("Transcribing audio for each speaker...")
    model = whisper.load_model("base")
    transcriptions = {}
    
    for speaker, audio_path in audio_paths.items():
        logger.info(f"Transcribing {speaker}...")
        result = model.transcribe(audio_path)
        transcriptions[speaker] = result["text"]
        logger.info(f"{speaker}: {result['text']}")
    
    return transcriptions


def save_transcriptions(transcriptions, output_dir):
    """Save transcriptions to file"""
    output_path = os.path.join(output_dir, "transcriptions.txt")
    with open(output_path, "w", encoding="utf-8") as f:
        for speaker, text in transcriptions.items():
            f.write(f"{speaker}: {text}\n\n")
    logger.info(f"Saved transcriptions to {output_path}")
    return output_path


async def process_audio_task(job_id):
    """Background task to process audio file"""
    job = jobs[job_id]
    
    try:
        job.status = "processing"
        
        # Step 1: Convert to WAV if needed
        wav_path = convert_to_wav(job.audio_path, os.path.join(job.result_dir, "input.wav"))
        
        # Step 2: Perform diarization
        speaker_segments = perform_diarization(wav_path, job.hf_token)
        
        # Step 3: Extract speaker audio
        job.speaker_audio_paths = extract_speaker_audio(wav_path, speaker_segments, job.result_dir)
        
        # Step 4: Transcribe
        job.transcriptions = transcribe_audio(job.speaker_audio_paths)
        
        # Step 5: Save transcriptions
        save_transcriptions(job.transcriptions, job.result_dir)
        
        job.status = "completed"
        logger.info(f"Job {job_id} completed successfully")
    except Exception as e:
        job.status = "failed"
        job.error = str(e)
        logger.error(f"Job {job_id} failed: {str(e)}")


@app.get("/")
async def root(request: Request):
    """Serve the main page"""
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/process/")
async def process_audio(background_tasks: BackgroundTasks, 
                        audio_file: UploadFile = File(...), 
                        hf_token: Optional[str] = Form(None)):
    """Process audio file and return job ID"""
    try:
        # Generate job ID
        job_id = str(uuid.uuid4())
        
        # Create job directory
        job_dir = os.path.join(UPLOAD_DIR, job_id)
        os.makedirs(job_dir, exist_ok=True)
        
        # Save uploaded file
        file_path = os.path.join(job_dir, audio_file.filename)
        with open(file_path, "wb") as f:
            shutil.copyfileobj(audio_file.file, f)
        
        # Create job
        job = Job(job_id, file_path, hf_token)
        jobs[job_id] = job
        
        # Start processing in background
        background_tasks.add_task(process_audio_task, job_id)
        
        return {"job_id": job_id, "status": "pending"}
    except Exception as e:
        logger.error(f"Error processing upload: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/status/{job_id}")
async def get_job_status(job_id: str):
    """Get status of job"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return jobs[job_id].to_dict()


@app.get("/download/{job_id}/{filename}")
async def download_file(job_id: str, filename: str):
    """Download a file from job results"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    if job.status != "completed":
        raise HTTPException(status_code=400, detail="Job not completed")
    
    file_path = os.path.join(job.result_dir, filename)
    if not os.path.exists(file_path):
        # Check if it's a speaker file
        for speaker, path in job.speaker_audio_paths.items():
            if speaker == filename:
                file_path = path
                break
        
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(file_path, filename=filename)


@app.get("/jobs")
async def list_jobs():
    """List all jobs"""
    return {job_id: job.to_dict() for job_id, job in jobs.items()}


@app.delete("/jobs/{job_id}")
async def delete_job(job_id: str):
    """Delete a job and its files"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Delete job files
    job = jobs[job_id]
    job_upload_dir = os.path.join(UPLOAD_DIR, job_id)
    if os.path.exists(job_upload_dir):
        shutil.rmtree(job_upload_dir)
    
    if os.path.exists(job.result_dir):
        shutil.rmtree(job.result_dir)
    
    # Remove job from memory
    del jobs[job_id]
    
    return {"status": "deleted"}


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=10000, reload=True)