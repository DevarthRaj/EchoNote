from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import whisper
import torch
from transformers import pipeline
from pydub import AudioSegment
import os

app = FastAPI()

# Load Whisper model
print("Loading Whisper model...")
whisper_model = whisper.load_model("base")  # Use "medium" or "large" for better accuracy
print("Whisper model loaded.")

# Load summarization pipeline
print("Loading summarizer...")
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")  # Replace with fine-tuned if needed
print("Summarizer loaded.")


def convert_audio_to_wav(file_path: str) -> str:
    """
    Convert any audio format to WAV using pydub.
    Whisper expects WAV, MP3, etc.
    """
    audio = AudioSegment.from_file(file_path)
    wav_path = "temp.wav"
    audio.export(wav_path, format="wav")
    return wav_path


@app.post("/transcribe/")
async def transcribe_and_summarize(file: UploadFile = File(...)):
    """
    Endpoint: Upload audio → returns transcription + summary
    """
    # Save the uploaded file
    input_path = f"temp_input.{file.filename.split('.')[-1]}"
    with open(input_path, "wb") as buffer:
        buffer.write(await file.read())

    # Convert to WAV
    wav_path = convert_audio_to_wav(input_path)

    # Transcribe using Whisper
    print("Transcribing...")
    result = whisper_model.transcribe(wav_path, language="ml")
    transcription = result["text"]
    print("Transcription done.")

    # Summarize using BART
    print("Summarizing...")
    summary = summarizer(transcription, max_length=100, min_length=30, do_sample=False)[0]['summary_text']
    print("Summary done.")

    # Clean up temp files
    os.remove(input_path)
    os.remove(wav_path)

    return JSONResponse(content={
        "transcription": transcription,
        "summary": summary
    })
