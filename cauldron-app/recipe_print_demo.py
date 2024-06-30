import sounddevice as sd
import scipy.io.wavfile as wav
import requests
import os
from thermal_printer_util import printer
from dotenv import load_dotenv
import numpy as np
from rpi_app_request import app_request
from time import sleep
from neopixel_util import *
from gpiozero import Button
from logging_util import logger


# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# GPIO setup
button_pin = 24
button = Button(button_pin, pull_up=True)

# Audio recording parameters
SAMPLE_RATE = 44100  # Sample rate
CHANNELS = 1  # Number of audio channels
FILENAME = "recording.wav"  # Filename for the recorded audio

# Global Idle Flag
global _idle_flag
_idle_flag = True

# Initialize recording list
recording = []

# Function to record audio
def start_recording():
    global recording
    global _idle_flag
    if not _idle_flag:
        logger.info("Recording aborted, _idle_flag is False")
        return
    
    logger.info("Recording started...")
    recording = []  # Reset the recording list

    # Start the audio stream
    stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS)
    stream.start()

    # Record audio chunks while the button is pressed
    while button.is_pressed:
        data, _ = stream.read(1024)
        recording.append(data)

    stream.stop()
    stream.close()
    logger.info("Recording finished")

    if recording:
        recording = np.concatenate(recording, axis=0)  # Concatenate audio chunks
        wav.write(FILENAME, SAMPLE_RATE, recording)
    else:
        logger.info("No audio data recorded.")

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

def recipe_header():
    os.system("lp -o orientation-requested=3 CALDRON_RECIPE_HEADER.bmp")
    sleep(10)

def recipe_footer():
    os.system("lp -o orientation-requested=3 RECIPE_FOOTER.bmp")
    sleep(10)

# Button callback
def button_pressed():
    logger.info("Button pressed! Starting recording...")
    global _idle_flag 
    _idle_flag = False
    start_recording()
    if len(recording) > 0:
        text = transcribe_audio(FILENAME)
        logger.info(f"Transcribed Text: {text}")
        
        recipe_header()

        if text != None:
            # Pass to Caldron App
            highlight_section('Tavily')
            try:
                app_request(text)
            except Exception as e:
                logger.error(e)

        recipe_footer()
        
    else:
        logger.info("No audio recorded. Skipping transcription and printing.")
    
    _idle_flag = True

# Attach the button callback
button.when_pressed = button_pressed

try:
    # Keep the script running
    logger.info(''.join(["Printer has paper? :", str(printer.has_paper())]))
    while True:
        while _idle_flag:
            color_wipe()
except KeyboardInterrupt:
    logger.info("Exiting program")
finally:
    pass  # gpiozero handles GPIO cleanup
