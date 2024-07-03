import sounddevice as sd
import scipy.io.wavfile as wav
import requests
import os
from thermal_printer_util import printer
from dotenv import load_dotenv
import numpy as np
from rpi_app_request import app_request
from time import sleep, time
from neopixel_util import *
from gpiozero import Button
from logging_util import logger
from twilio_util import send_alert
from cauldron_app import CaldronApp

last_msg_time = None
app = CaldronApp("gpt-3.5-turbo", verbose=True)

def alert(message):
    global last_msg_time
    current_time = time.time()
    # Check if one hour has passed since the last SMS
    if (last_msg_time == None) or current_time - last_msg_time >= 3600:
        send_alert(message)
        last_msg_time = current_time
        logger.info("WhatsApp sent: " + message)
    else:
        logger.info("WhatsApp not sent, waiting for cooldown period.")

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# GPIO setup
button_pin = 24
button = Button(button_pin, pull_up=True, bounce_time=0.5)

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
    
    logger.info("Recording started...")
    recording = []  # Reset the recording list
    color_wipe()

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
    logger.info("Button pressed!")
    global _idle_flag
        
    logger.info("Starting recording...")
    start_recording()
    _idle_flag = False
    
    ##CORE APP FUNCTION
    if len(recording) > 0:
        text = transcribe_audio(FILENAME)
        logger.info(f"Transcribed Text: {text}")
        
        #check to make sure there's printer paper
        if not printer.has_paper():
            logger.error("Printer is out of paper. Please refill the paper tray.")
            alert("The printer is out of paper. Please refill the paper tray.")
            #blink the light ring red
            return
        
        recipe_header()

        if text != None:
            # Pass to Caldron App
            try:
                app_request(app, text)
            except Exception as e:
                logger.error(e)

        clear_ring()
        recipe_footer()
        
    else:
        logger.info("No audio recorded. Skipping transcription and printing.")
    
    rotate_brightness()
    _idle_flag = True

# Attach the button callback
button.when_pressed = button_pressed

try:
    # Keep the script running
    if printer.has_paper():
        logger.info(''.join(["Printer has paper? :", str(printer.has_paper())]))
        rotate_brightness()
        while True:
            sleep(0.25)
    else:
        logger.error("Printer is out of paper. Please refill the paper tray.")
        #blink the light ring red
except KeyboardInterrupt:
    logger.info("Exiting program")
finally:
    pass  # gpiozero handles GPIO cleanup
