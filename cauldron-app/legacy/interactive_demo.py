import tkinter as tk
from tkinter.scrolledtext import ScrolledText
import serial
import threading
import time
from util import VoiceInterface, find_SQL_prompt
from logging_util import logger

class InteractiveDemo:
    def __init__(self, master, app):
        
        logger.info("Initializing Functional Demo Application")
        
        self.master = master
        self.app = app
        self.running = True  # Flag to indicate that the app is running
        self.master.title("Cauldron Functional Demo Application")
        self.master.geometry("1200x600")  # Adjust size as necessary

        # Constants for serial communication
        self.SERIAL_PORT = 'COM4'
        self.BAUD_RATE = 115200
        self.INITIAL_PERIOD = 1
        self.ADC_BOUND = 7
        self.start_time = time.time()
        logger.info("Serial Port and ADC Constants Initialized")

        # Voice interface
        self.voice_interface = VoiceInterface()
        logger.info("Voice Interface Initialized")

         # Set up the grid layout for the main window
        master.grid_columnconfigure(0, weight=1)
        master.grid_columnconfigure(1, weight=1)
        master.grid_rowconfigure(0, weight=1)
        master.grid_rowconfigure(1, weight=1)
        logger.info("Grid Layout Initialized")

        # Setup the text input and response area in the top-left and bottom-left quadrants
        self.frame_text_interaction = tk.Frame(master)
        self.frame_text_interaction.grid(row=0, column=0, sticky='nsew', rowspan=2)  # Span two rows

        self.input_text = tk.Entry(self.frame_text_interaction)
        self.input_text.pack(fill=tk.X, padx=10, pady=10)

        self.output_text = ScrolledText(self.frame_text_interaction)
        self.output_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.send_button = tk.Button(self.frame_text_interaction, text="Send", command=self.process_input)
        self.send_button.pack(fill=tk.X, padx=10, pady=5)

        self.clear_button = tk.Button(self.frame_text_interaction, text="Clear", command=self.clear_output)
        self.clear_button.pack(fill=tk.X, padx=10, pady=5)
        logger.info("Text Interaction Components Initialized")

        # Transcription display in the upper right quadrant
        self.transcription_text = ScrolledText(master, height=10)
        self.transcription_text.grid(row=0, column=1, sticky='nsew', padx=10, pady=10)

        # Recording Indicator
        self.canvas = tk.Canvas(master, width=20, height=20)
        self.recording_indicator = self.canvas.create_oval(5, 5, 20, 20, fill='red')
        self.canvas.itemconfig(self.recording_indicator, state='hidden')  # Initially hidden
        self.canvas.grid(row=0, column=1, sticky='ne', padx=10, pady=10)

        self.record_button = tk.Button(master, text="Record Audio", command=self.start_recording)
        self.record_button.grid(row=0, column=1, sticky='n', padx=10, pady=(50, 0))

        self.transcribe_button = tk.Button(master, text="Transcribe Audio", command=self.transcribe_audio)
        self.transcribe_button.grid(row=0, column=1, sticky='n', padx=10, pady=(90, 0))

        self.copy_to_chat_button = tk.Button(master, text="Copy to Chat Input", command=self.copy_transcription_to_chat)
        self.copy_to_chat_button.grid(row=0, column=1, sticky='n', padx=10, pady=(130, 0))
        logger.info("Transcription Components Initialized")

        # Sensor display components
        self.lights = []
        self.adc_texts = []
        self.initial_readings = [[] for _ in range(8)]
        self.setup_sensor_display(master)

        # Serial connection
        try:
            self.ser = serial.Serial(self.SERIAL_PORT, self.BAUD_RATE)
            self.start_time = time.time()
            self.baseline = [0] * 8
            self.initial_readings = [[] for _ in range(8)]
            self.start_sensor_reading()
            logger.info("Serial connection established.")
        except serial.SerialException as e:
            logger.error("Error opening serial port: %s", e, exc_info=False)
            logger.warning("Serial connection could not be established.")

    def start_recording(self):
        """Start the recording in a separate thread to avoid UI blocking."""
        def hide_indicator():
            self.canvas.itemconfig(self.recording_indicator, state='hidden')  # Hide indicator

        self.canvas.itemconfig(self.recording_indicator, state='normal')  # Show indicator
        self.master.after(100, lambda: threading.Thread(target=lambda: self.voice_interface.record_audio(callback=hide_indicator)).start())

    def transcribe_audio(self):
        """Transcribe the recorded audio and display it."""
        transcription = self.voice_interface.transcribe_audio()
        self.transcription_text.delete("1.0", tk.END)  # Clear all text from the text box
        self.transcription_text.insert(tk.END, transcription + "\n")
        self.transcription_text.see(tk.END)

    def copy_transcription_to_chat(self):
        """
        Copies the contents of the transcription text box to the chatbot input text box.
        """
        # Get text from transcription_text ScrolledText widget
        transcription_text = self.transcription_text.get("1.0", tk.END).strip()

        # Clear existing content in the input_text Entry widget and insert the new text
        self.input_text.delete(0, tk.END)
        self.input_text.insert(0, transcription_text)

    def setup_sensor_display(self, parent):
        # Frame for sensor display in the lower right quadrant
        sensor_frame = tk.Frame(parent)
        sensor_frame.grid(row=1, column=1, sticky='nsew')
        
        # Configuring a 5x5 grid
        for i in range(5):
            sensor_frame.grid_rowconfigure(i, weight=1)
            sensor_frame.grid_columnconfigure(i, weight=1)

        # Specifying the positions of each sensor within the 5x5 grid
        sensor_positions = {
            0: (1, 2),
            1: (0, 2),
            2: (2, 1), 
            3: (2, 0), 
            4: (4, 2), 
            5: (3, 2),
            6: (2, 3),
            7: (2, 4)
        }

        # Creating sensor displays
        for sensor_id, (row, col) in sensor_positions.items():
            frame = tk.Frame(sensor_frame, bd=2, relief="ridge")
            frame.grid(row=row, column=col, padx=10, pady=10, sticky='nsew')

            light_canvas = tk.Canvas(frame, bg="red", height=60, width=60)
            light_canvas.pack(fill='both', expand=True)
            
            adc_text = light_canvas.create_text(30, 30, text=f"ADC: 0", fill="white")
            self.adc_texts.append(adc_text)
            self.lights.append(light_canvas)

        # Add reset baseline button
        reset_button = tk.Button(sensor_frame, text="Reset Baseline", command=self.reset_baseline)
        reset_button.grid(row=4, column=4, padx=10, pady=10, sticky='nsew')  # Positioned at the bottom right

    def start_sensor_reading(self):
        threading.Thread(target=self.check_serial, daemon=True).start()

    def reset_baseline(self):
        """Reset the baseline values to current ADC readings."""
        for i in range(8):
            if self.initial_readings[i]:
                self.baseline[i] = sum(self.initial_readings[i]) / len(self.initial_readings[i])
                print(f"Sensor {i} baseline reset to {self.baseline[i]}")

    def check_serial(self):
        while self.running:
            current_time = time.time()
            if self.ser.in_waiting:
                line = self.ser.readline()
                try:
                    decoded_line = line.decode('utf-8').strip().split('\t')
                    for i in range(8):

                        current_value = int(decoded_line[i * 3 + 1])

                        if current_time - self.start_time <= self.INITIAL_PERIOD:
                            if current_value != 0:
                                self.initial_readings[i].append(current_value)
                                self.baseline[i] = sum(self.initial_readings[i]) / len(self.initial_readings[i])
                        
                        self.lights[i].itemconfig(self.adc_texts[i], text=f"ADC: {current_value}")
                        color = self.calculate_color(current_value, self.baseline[i])
                        self.lights[i].config(bg=color)
                        
                except Exception as e:
                    print("Error:", e)
            time.sleep(0.01)

    def calculate_color(self, value, baseline):
        difference = abs(value - baseline)
        if difference < self.ADC_BOUND:
            return "#ff0000"  # Red for values within the bound
        else:
            green_intensity = int(max(0, 255 - int((difference - self.ADC_BOUND))) * 0.5)
            return f"#00{green_intensity:02x}00"

    def append_output_chat(self,chat_output):
        if "actions" in chat_output:
            pass
            #for action in chat_output["actions"]:
            #    self.output_text.insert(tk.END, f"Calling Tool: `{action.tool}` with input `{action.tool_input}`\n")
        # Observation
        elif "steps" in chat_output:
            pass
            #for step in chat_output["steps"]:
            #    self.output_text.insert(tk.END, f"Tool Result: `{step.observation}`\n")
        # Final result
        elif "output" in chat_output:
            self.output_text.insert(tk.END, f'Final Output: {chat_output["output"]}\n')
        else:
            self.output_text.insert(tk.END, chat_output)

        self.output_text.see(tk.END)

    def finalize_stream(self):
        self.execute_sql_commands()  # Execute collected SQL commands

    def process_input(self):
        user_input = self.input_text.get()
        if user_input.lower() == "exit":
            self.output_text.insert(tk.END, "Exiting...\n")
            self.master.quit()  # Close the application
        else:
            # Start streaming in a separate thread, properly passing the callback and on_complete functions
            thread = threading.Thread(target=lambda: self.app.UIAgent.stream(user_input, self.append_output_chat, self.finalize_stream))
            thread.start()
            self.input_text.delete(0, tk.END)

    def execute_sql_commands(self):
        self.sqlQueries = find_SQL_prompt(self.output_text.get("1.0", "end-1c"))
        self.process_queries_sequentially()

    def process_queries_sequentially(self):
        if self.sqlQueries:
            query = self.sqlQueries.pop(0)
            thread = threading.Thread(target=self.run_query_and_append, args=(query,))
            thread.start()
            thread.join()  # Wait for the thread to complete
            nil=input("Pause for demo...")

    def run_query_and_append(self, query):
        self.app.SQLAgent.stream(query, self.append_output_chat)
        return

    def clear_output(self):
        self.output_text.delete("1.0", tk.END)  # Clear all text from the output text box

    def on_close(self):
        """Clean up and close the application gracefully."""
        self.running = False  # Signal all threads to stop
        self.master.after(500, self.master.destroy)  # Delay to allow threads to terminate

        # Additional cleanup if necessary, like closing serial connections:
        if hasattr(self, 'ser') and self.ser.is_open:
            self.ser.close()       



