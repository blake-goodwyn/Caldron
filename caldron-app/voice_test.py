from pydub import AudioSegment
from pydub.silence import detect_nonsilent
from pyaudio import PyAudio, paInt16
import wave
import os
from dotenv import load_dotenv
from openai import OpenAI
import numpy as np
from collections import deque

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# Audio recording parameters
THRESHOLD = 30  # dB, adjust according to your needs
CHUNK = 2048
FORMAT = paInt16
CHANNELS = 1
RATE = 44100
SILENCE_DURATION = 1300  # ms
INITIAL_SILENCE_DURATION = 10000  # ms
MOVING_AVERAGE_WINDOW = 10
DEVICE_INDEX = 2  # Set the index of the desired microphone
output_filename = "voice_input.wav"

# Initialize PyAudio
audio = PyAudio()

# Function to calculate decibel level
def calculate_db(data):
    rms = np.sqrt(np.mean(np.square(np.frombuffer(data, dtype=np.int16))))
    return 20 * np.log10(rms + 1e-10)

# Function to record audio
def record_audio():
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True, input_device_index=DEVICE_INDEX,
                        frames_per_buffer=CHUNK)
    print("Waiting for voice input...")
    frames = []
    silent_frames = 0
    db_levels = deque(maxlen=MOVING_AVERAGE_WINDOW)

    while True:
        data = stream.read(CHUNK)
        db_level = calculate_db(data)
        db_levels.append(db_level)
        smoothed_db_level = np.mean(db_levels) if len(db_levels) == MOVING_AVERAGE_WINDOW else 0
        #print(smoothed_db_level)
        frames.append(data)
        audio_segment = AudioSegment(
            data=b''.join(frames), sample_width=audio.get_sample_size(FORMAT),
            frame_rate=RATE, channels=CHANNELS)

        # Detect non-silent chunks
        nonsilent_chunks = detect_nonsilent(audio_segment, min_silence_len=INITIAL_SILENCE_DURATION, silence_thresh=THRESHOLD)

        if nonsilent_chunks:
            break

    print("Voice detected. Recording...")
    frames = []  # Reset frames to start fresh recording

    while True:
        data = stream.read(CHUNK)
        db_level = calculate_db(data)
        db_levels.append(db_level)
        smoothed_db_level = np.mean(db_levels) if len(db_levels) == MOVING_AVERAGE_WINDOW else 0
        #print(smoothed_db_level)
        frames.append(data)
        audio_segment = AudioSegment(
            data=b''.join(frames), sample_width=audio.get_sample_size(FORMAT),
            frame_rate=RATE, channels=CHANNELS)

        # Detect non-silent chunks
        nonsilent_chunks = detect_nonsilent(audio_segment, min_silence_len=SILENCE_DURATION, silence_thresh=THRESHOLD)

        if not nonsilent_chunks:
            silent_frames += 1
        else:
            silent_frames = 0

        if silent_frames > (RATE / CHUNK) * (SILENCE_DURATION / 1000):
            # If prolonged silence is detected, stop recording
            break

    print("Recording stopped")
    stream.stop_stream()
    stream.close()
    return b''.join(frames)

# Save the recorded audio to a file
def save_audio(filename, audio_data):
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(audio_data)

# Transcribe audio using Whisper API
def transcribe_audio(filename):
    with open(filename, "rb") as audio_file:
        translation = client.audio.translations.create(
            model="whisper-1", 
            file=audio_file
        )
    return translation.text

# Main function
def main():
    audio_data = record_audio()
    save_audio(output_filename, audio_data)
    transcript = transcribe_audio(output_filename)
    print("Transcription:", transcript)

if __name__ == "__main__":
    main()