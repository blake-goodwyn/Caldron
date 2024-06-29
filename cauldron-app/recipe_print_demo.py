import sounddevice as sd
import scipy.io.wavfile as wav
import requests
import os
import serial
import adafruit_thermal_printer
from dotenv import load_dotenv
import numpy as np
from cauldron_app import CaldronApp
from class_defs import Recipe, Ingredient, load_graph_from_file, load_pot_from_file
from langchain_util import quickTextChain
from custom_print import printer as pretty
from time import sleep
from neopixel_util import *
from gpiozero import Button
from logging_util import logger
import textwrap

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
    try:
        printer.print(text)
    finally:
        printer.feed(3)

# Button callback
def button_pressed():
    logger.info("Button pressed! Starting recording...")
    global _idle_flag 
    _idle_flag = False
    start_recording()
    if len(recording) > 0:
        text = transcribe_audio(FILENAME)
        logger.info(f"Transcribed Text: {text}")
        
        os.system("lp -o orientation-requested=3 CALDRON_RECIPE_HEADER.bmp")

        msg = quickTextChain.invoke({'input': text})
        printer.print(msg.content)

        # Pass to Caldron App
        highlight_section('Tavily')
        app.post(text)
        while app.printer_wait_flag:
            sleep(0.5)
        
        recipe_graph = load_graph_from_file(app.recipe_graph_file)
        recipe = recipe_graph.get_foundational_recipe()
        if (recipe != None):
            pot = load_pot_from_file(app.recipe_pot_file)
            recipe = pot.pop_recipe()
            if (recipe != None):
                try:
                    print_recipe(pretty.format(recipe))
                except Exception as e:
                    logger.error(e)
            else:
                print_recipe(textwrap.fill("Sorry, I couldn't find an appropriate recipe for what you were looking for.", width=32))
        else:
            try:
                print_recipe(pretty.format(recipe))
            except Exception as e:
                logger.error(e)
        
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
