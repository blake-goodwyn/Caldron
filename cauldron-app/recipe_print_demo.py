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
from class_defs import Recipe, Ingredient
from custom_print import printer as pretty
from time import sleep
import board
import neopixel
from gpiozero import Button

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
    print("Recording started...")
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
    print("Recording finished")

    if recording:
        recording = np.concatenate(recording, axis=0)  # Concatenate audio chunks
        wav.write(FILENAME, SAMPLE_RATE, recording)
    else:
        print("No audio data recorded.")

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
    display_neopixel()  # Display Neopixel lights before printing
    os.system("lp -o orientation-requested=3 CALDRON_RECIPE_HEADER.bmp")
    sleep(10)
    wrapped_text = wrap_text(text, 32)  # Wrap text to 32 characters per line
    printer.print(wrapped_text)
    printer.feed(3)

# Function to wrap text for the thermal printer
def wrap_text(text, width):
    wrapped_lines = textwrap.fill(text, width)
    return wrapped_lines

# Function to display Neopixel lights
def display_neopixel():
    for i in range(num_pixels):
        pixels[i] = (255, 0, 0)  # Red color
    pixels.show()
    sleep(3)
    pixels.fill((0, 0, 0))  # Turn off the lights
    pixels.show()

# Button callback
def button_pressed():
    print("Button pressed! Starting recording...")
    start_recording()
    if len(recording) > 0:
        text = transcribe_audio(FILENAME)
        print(f"Transcribed Text: {text}")
        print_recipe(text)
    else:
        print("No audio recorded. Skipping transcription and printing.")

# Attach the button callback
button.when_pressed = button_pressed

try:
    # Keep the script running
    print("Printer has paper? :", printer.has_paper())
    while True:
        sleep(1)
except KeyboardInterrupt:
    print("Exiting program")
finally:
    pass  # gpiozero handles GPIO cleanup
