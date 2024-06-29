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
app = CaldronApp('gpt-3.5-turbo')

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

waiting_color = (255, 178, 0)

def color_wipe(max_brightness=0.2, wait_ms=1500):
    """ Smoothly wipe brightness across the NeoPixel ring """
    steps = int(wait_ms / 100)  # Number of steps to achieve smooth transition
    brightness_step = max_brightness / steps
    
    for i in range(steps + 1):
        current_brightness = brightness_step * i
        pixels.fill((int(waiting_color[0] * current_brightness), 
                     int(waiting_color[1] * current_brightness), 
                     int(waiting_color[2] * current_brightness)))
        pixels.show()
        time.sleep(wait_ms / (steps * 1000))  # Adjust to keep the same total time
        
    # Decreasing brightness
    for i in range(steps, -1, -1):
        current_brightness = brightness_step * i
        pixels.fill((int(waiting_color[0] * current_brightness), 
                     int(waiting_color[1] * current_brightness), 
                     int(waiting_color[2] * current_brightness)))
        pixels.show()
        time.sleep(wait_ms / (steps * 1000))  # Adjust to keep the same total time
    
    pixels.fill((0, 0, 0))  # Turn off all pixels after animation
    pixels.show()

def clear_ring():
    pixels.fill((0, 0, 0))  # Turn off all pixels after animation
    pixels.show()
    
def highlight_section(agent):
    
    agent_locs = {
        'Tavily': [20, 21, 22],
        'Frontman': [5,6,7],
        'Spinnaret': [10,11,12],
        'Sleuth': [15,16,17]
    }
    
    clear_ring()
    if agent in agent_locs.keys():
        for i in agent_locs[agent]:
            pixels[i] = waiting_color
        pixels.show()
    
    

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
