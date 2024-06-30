import requests
from dotenv import load_dotenv
import os
import sounddevice as sd
import scipy.io.wavfile as wav
from time import sleep, time
import keyboard
from rpi_app_request import app_request

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Function to list available audio devices
def list_audio_devices():
    print("Available audio devices:")
    devices = sd.query_devices()
    for i, device in enumerate(devices):
        print(f"{i}: {device['name']} - {'input' if device['max_input_channels'] > 0 else 'output'}")
    return devices

# Function to record audio
def record_audio(filename, duration=5, fs=44100, device=None):
    print("Recording...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=2, dtype='int16', device=device)
    sd.wait()  # Wait until recording is finished
    wav.write(filename, fs, recording)
    print(f"Recording saved as {filename}")

# Function to transcribe audio using Whisper API
def transcribe_audio(filename):
    with open(filename, "rb") as f:
        response = requests.post(
            "https://api.openai.com/v1/audio/transcriptions",
            headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}"
            },
            files={"file": f},
            data={"model": "whisper-1"}
        )
    return response.json().get("text", "")

# List available devices and select one
#devices = list_audio_devices()
device_index = 2

while True:
    print("Press space bar to start recording...")
    
    # Wait for space bar to be pressed
    keyboard.wait('space')
    
    # Record audio
    audio_filename = "recorded_audio.wav"
    record_audio(audio_filename, device=device_index)
    
    # Transcribe audio
    transcription = transcribe_audio(audio_filename)
    print(f"Transcription: {transcription}")
    
    app_request(transcription)

    # Small delay to prevent accidental double recording
    sleep(1)
