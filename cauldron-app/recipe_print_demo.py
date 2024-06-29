import sounddevice as sd
import scipy.io.wavfile as wav
import requests
import os
import serial
import adafruit_thermal_printer
from dotenv import load_dotenv
import numpy as np
import textwrap
from cauldron_app import CaldronApp
from class_defs import Recipe, Ingredient, load_graph_from_file
from custom_print import printer as pretty
from time import sleep
import board
import math
import time
import threading
import neopixel
from gpiozero import Button
from logging_util import logger

# Example recipe
demo_recipe = Recipe(
    name="Chocolate Chip Cookies",
    ingredients=[Ingredient(name="cookies", quantity=6, unit="unit")],
    instructions=["Make them."]
)

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# GPIO setup
button_pin = 24
button = Button(button_pin, pull_up=True)

#Caldron App
app = CaldronApp()

# Neopixel setup
pixel_pin = board.D18
num_pixels = 24
ORDER = neopixel.RGB
pixels = neopixel.NeoPixel(pixel_pin, num_pixels, brightness=0.2, auto_write=False, pixel_order=ORDER)

# Thermal printer setup (adjust to your specific setup)
uart = serial.Serial("/dev/serial0", baudrate=9600, timeout=3000)
ThermalPrinter = adafruit_thermal_printer.get_printer_class(2.69)
printer = ThermalPrinter(uart)
header_img = 'CALDRON_RECIPE_HEADER.bmp'
printer.size = adafruit_thermal_printer.SIZE_SMALL
printer.justify = adafruit_thermal_printer.JUSTIFY_LEFT
printer.underline = None

# Audio recording parameters
SAMPLE_RATE = 44100  # Sample rate
CHANNELS = 1  # Number of audio channels
FILENAME = "recording.wav"  # Filename for the recorded audio

# Initialize recording list
recording = []

# Function to record audio
def start_recording():
    global recording
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

def print_recipe(text):
    os.system("lp -o orientation-requested=3 CALDRON_RECIPE_HEADER.bmp")
    sleep(10)
    wrapped_text = wrap_text(text, 32)  # Wrap text to 32 characters per line
    printer.logger.info(wrapped_text)
    printer.feed(3)

# Function to wrap text for the thermal printer
def wrap_text(text, width):
    wrapped_lines = textwrap.fill(text, width)
    return wrapped_lines

waiting_color = (255, 209, 102)  # RGB color (255, 209, 102) -> Orange

def breathing_animation(cycle_duration=1.5, steps=100):
    """ Displays a breathing animation on the NeoPixel ring """
    while not button_pressed:  # Continue animation until button is pressed
        for i in range(steps):
            brightness = 0.5 + 0.5 * math.sin(i / steps * 2 * math.pi)
            pixels.fill((int(waiting_color[0] * brightness), int(waiting_color[1] * brightness), int(waiting_color[2] * brightness)))
            pixels.show()
            time.sleep(cycle_duration / steps)

# Button callback
def button_pressed():
    logger.info("Button pressed! Starting recording...")
    start_recording()
    if len(recording) > 0:
        text = transcribe_audio(FILENAME)
        logger.info(f"Transcribed Text: {text}")

        # Pass to Caldron App
        app.post(text)
        while app.printer_wait_flag:
            sleep(1)
        
        logger.debug("Getting foundational recipe from recipe graph.")
        recipe_graph = load_graph_from_file(app.recipe_graph_file)
        recipe = recipe_graph.get_foundational_recipe()
        print_recipe(pretty(recipe))
    else:
        logger.info("No audio recorded. Skipping transcription and printing.")

# Attach the button callback
button.when_pressed = button_pressed

try:
    # Keep the script running
    logger.info("Printer has paper? :", printer.has_paper())
    breathing_thread = threading.Thread(target=breathing_animation)
    breathing_thread.start()
    while True:
        sleep(1)
except KeyboardInterrupt:
    logger.info("Exiting program")
finally:
    pass  # gpiozero handles GPIO cleanup
