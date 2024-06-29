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
from neopixel_util import *
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
app = CaldronApp('gpt-3.5-turbo')

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

#Global Idle Flag
global _idle_flag
_idle_flag = True

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
    printer.print(wrapped_text)
    printer.feed(3)

# Function to wrap text for the thermal printer
def wrap_text(text, width):
    wrapped_lines = textwrap.fill(text, width)
    return wrapped_lines

# Button callback
def button_pressed():
    logger.info("Button pressed! Starting recording...")
    global _idle_flag 
    _idle_flag = False
    start_recording()
    if len(recording) > 0:
        text = transcribe_audio(FILENAME)
        logger.info(f"Transcribed Text: {text}")

        # Pass to Caldron App
        app.post(text)
        while app.printer_wait_flag:
            sleep(0.5)
        
        logger.debug("Getting foundational recipe from recipe graph.")
        recipe_graph = load_graph_from_file(app.recipe_graph_file)
        recipe = recipe_graph.get_foundational_recipe()
        if (recipe != None):
            print_recipe(pretty.pformat(recipe))
        else:
            print_recipe("Sorry, I couldn't find an appropriate recipe for what you were looking for.")
        
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
