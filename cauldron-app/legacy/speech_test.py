import tkinter as tk
from tkinter import messagebox
import sounddevice as sd
import soundfile as sf
from openai import OpenAI
from dotenv import load_dotenv
import os
import threading

# Load environment variables
load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')

# Initialize OpenAI client
client = OpenAI(api_key=api_key)

# Audio recording parameters
samplerate = 44100  # Sample rate in Hertz
duration = 5  # Duration of recording in seconds
filename = 'output.wav'  # File name to save the audio

def record_audio(mic_index=None):
    """ Record audio from the specified microphone """
    global recording
    recording = True
    update_recording_indicator(True)  # Show the red circle
    myrecording = sd.rec(int(samplerate * duration), samplerate=samplerate, channels=2, dtype='float64', device=mic_index)
    sd.wait()  # Wait until recording is finished
    sf.write(filename, myrecording, samplerate)
    recording = False
    update_recording_indicator(False)  # Hide the red circle

def transcribe_audio():
    """ Transcribe audio file using OpenAI API """
    with open(filename, "rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file
        )
    return transcription.text

def start_recording():
    """ Start recording in a separate thread to avoid freezing the GUI """
    mic_index = mic_names.index(selected_mic.get())
    threading.Thread(target=lambda: record_audio(mic_index=mic_index)).start()

def update_recording_indicator(show):
    """ Update the recording indicator on the GUI """
    if show:
        canvas.itemconfig(recording_indicator, state='normal')  # Show the red circle
    else:
        canvas.itemconfig(recording_indicator, state='hidden')  # Hide the red circle

def start_transcription():
    """ Handle transcription of recorded audio """
    if not recording:
        text = transcribe_audio()
        text_box.delete('1.0', tk.END)
        text_box.insert(tk.END, text)
    else:
        messagebox.showerror("Error", "Recording is still in progress. Please wait.")

# Set up the GUI
root = tk.Tk()
root.title("Audio Recorder and Transcriber")

# Dropdown menu for microphone selection
mic_names = [device['name'] for device in sd.query_devices() if device['max_input_channels'] > 0]
selected_mic = tk.StringVar(root)
selected_mic.set(mic_names[0])  # default value
mic_menu = tk.OptionMenu(root, selected_mic, *mic_names)
mic_menu.pack(pady=10)

# Canvas for recording indicator
canvas = tk.Canvas(root, width=20, height=20)
recording_indicator = canvas.create_oval(5, 5, 20, 20, fill='red')
canvas.itemconfig(recording_indicator, state='hidden')  # Initially hidden
canvas.pack(side=tk.LEFT, padx=10)

# Create a button to start recording
record_button = tk.Button(root, text="Record Audio", command=start_recording)
record_button.pack(side=tk.LEFT, pady=20)

# Create a button to transcribe audio
transcribe_button = tk.Button(root, text="Transcribe Audio", command=start_transcription)
transcribe_button.pack(pady=20)

# Text box to display the transcription
text_box = tk.Text(root, height=10, width=50)
text_box.pack(pady=20)

root.mainloop()