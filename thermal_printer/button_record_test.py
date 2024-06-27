import RPi.GPIO as GPIO
import sounddevice as sd
import scipy.io.wavfile as wav
import requests
import os
import serial
import adafruit_thermal_printer
from dotenv import load_dotenv
import numpy as np
import textwrap

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# GPIO setup
GPIO.setmode(GPIO.BCM)
button_pin = 24
GPIO.setup(button_pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)

# Thermal printer setup (adjust to your specific setup)
uart = serial.Serial("/dev/serial0", baudrate=9600, timeout=3000)
ThermalPrinter = adafruit_thermal_printer.get_printer_class(2.69)
printer = ThermalPrinter(uart)

# Audio recording parameters
SAMPLE_RATE = 44100  # Sample rate
CHANNELS = 1  # Number of audio channels
FILENAME = "recording.wav"  # Filename for the recorded audio

# Function to record audio while button is pressed
def record_audio(filename):
    print("Recording...")
    recording = []  # List to hold audio chunks

    # Start the audio stream
    stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS)
    stream.start()

    try:
        while GPIO.input(button_pin) == GPIO.LOW:  # Record while button is pressed
            data, _ = stream.read(1024)
            recording.append(data)
    finally:
        stream.stop()
        stream.close()

    recording = np.concatenate(recording, axis=0)  # Concatenate audio chunks
    wav.write(filename, SAMPLE_RATE, recording)
    print("Recording finished")

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
    printer.feed(2)
    printer.size = adafruit_thermal_printer.SIZE_MEDIUM
    printer.justify = adafruit_thermal_printer.JUSTIFY_CENTER
    printer.underline = adafruit_thermal_printer.UNDERLINE_THICK
    printer.print('Caldron')
    printer.feed(2)

# Function to wrap text for the thermal printer
def wrap_text(text, width):
    wrapped_lines = textwrap.fill(text, width)
    return wrapped_lines

# Button callback
def button_callback(channel):
    print("Button pressed! Starting recording...")
    record_audio(FILENAME)
    text = transcribe_audio(FILENAME)
    print(f"Transcribed Text: {text}")
    wrapped_text = wrap_text(text, 32)  # Wrap text to 32 characters per line
    printer.print(wrapped_text)
    printer.feed(3)

# Setup event detection
GPIO.add_event_detect(button_pin, GPIO.FALLING, callback=button_callback, bouncetime=300)

try:
    # Keep the script running
    print("Printer has paper? :", printer.has_paper())
    recipe_header()
    while True:
        pass
except KeyboardInterrupt:
    print("Exiting program")
finally:
    GPIO.cleanup()
