import sounddevice as sd
import soundfile as sf
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

class VoiceInterface:
    def __init__(self, api_key=OPENAI_API_KEY):
        self.client = OpenAI(api_key=api_key)
        self.samplerate = 44100  # Sample rate in Hertz
        self.duration = 5  # Duration of recording in seconds
        self.filename = 'output.wav'  # File name to save the audio
        self.recording = False  # Recording state
        self.mic_index = 2  # Selected microphone index
        self.mic_list = self.list_microphones()

    def list_microphones(self):
        """Return a list of available microphones with input capability."""
        devices = sd.query_devices()
        return [device for device in devices if device['max_input_channels'] > 0]

    def select_microphone(self, mic_index):
        """Select the microphone by index."""
        if mic_index < len(self.mic_list):
            self.mic_index = mic_index
            print(f"Microphone selected: {self.mic_list[mic_index]['name']}")
        else:
            print("Invalid microphone index")
        return

    def confirm_microphone(self):
        """Return the currently selected microphone."""
        if self.mic_index is not None:
            print(f"Currently selected microphone: {self.mic_list[self.mic_index]['name']}")
        else:
            print("No microphone selected")
        return

    def record_audio(self, callback):
        """Record audio from the selected microphone."""
        if self.mic_index is None:
            print("No microphone selected")
        else:
            self.recording = True
            myrecording = sd.rec(int(self.samplerate * self.duration), samplerate=self.samplerate, channels=2, dtype='float64', device=self.mic_index)
            sd.wait()
            sf.write(self.filename, myrecording, self.samplerate)
            self.recording = False
            if callback:
                callback()
        return

    def transcribe_audio(self):
        """Transcribe the recorded audio file using OpenAI API."""
        if not self.recording:
            with open(self.filename, "rb") as audio_file:
                transcription = self.client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file
                )
            return transcription.text
        else:
            return "Recording in progress, please wait"
