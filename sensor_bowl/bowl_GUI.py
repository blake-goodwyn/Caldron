import tkinter as tk
import serial
import threading
import time

# Set to your Arduino's serial port
SERIAL_PORT = 'COM4'
BAUD_RATE = 115200
INITIAL_PERIOD = 1  # Initial period in seconds to establish baseline
ADC_BOUND = 7  # Bound for divergence

# Create the main window
root = tk.Tk()
root.title("Channel ADC Values")
root.geometry("600x400")  # Starting window size

# Initialize lights for each channel
lights = []
adc_texts = []  # Store the text items for updating
initial_readings = [[] for _ in range(8)]
baseline = [0] * 8
start_time = time.time()

# Define the grid layout for the channels
layout_positions = {
    1: (0, 2),
    0: (1, 2),
    2: (2, 0), 3: (2, 1),
    6: (2, 3), 7: (2, 4),
    4: (3, 2),
    5: (4, 2)
}

def calculate_color(value, baseline):
    difference = abs(value - baseline)
    if difference < ADC_BOUND:
        return "#ff0000"  # Red
    else:
        green_intensity = 255 - min(int((difference - ADC_BOUND) * 10), 255)
        return f"#00{green_intensity:02x}00"

def on_resize(event):
    # Update text positions on canvas resize
    for light, adc_text in zip(lights, adc_texts):
        light.coords(adc_text, light.winfo_width() / 2, light.winfo_height() / 2)

for i in range(8):
    frame = tk.Frame(root, bd=2, relief="ridge")
    light_canvas = tk.Canvas(frame, bg="red")
    light_canvas.pack(fill='both', expand=True)
    frame.grid(row=layout_positions[i][0], column=layout_positions[i][1], padx=10, pady=10, sticky='nsew')

    # Create static text for channel number
    light_canvas.create_text(100, 10, text=f"Channel {i}", fill="white")

    # Create dynamic text for ADC value (initially set to 0)
    adc_text = light_canvas.create_text(100, 35, text="ADC: 0", fill="white")
    adc_texts.append(adc_text)

    lights.append(light_canvas)

# Configure row/column weights for centering and symmetry
for r in range(5):
    root.grid_rowconfigure(r, weight=1)
for c in range(5):
    root.grid_columnconfigure(c, weight=1)

# Bind the resize event
root.bind("<Configure>", on_resize)

# Open serial port
ser = serial.Serial(SERIAL_PORT, BAUD_RATE)

def check_serial():
    current_time = time.time()
    if ser.in_waiting:
        line = ser.readline()
        try:
            decoded_line = line.decode('utf-8').strip()
            parts = decoded_line.split('\t')  # Split the line by tabs

            # Ensure the line has the expected number of elements (3 elements per channel)
            if len(parts) == 36:  # 12 channels * 3 elements per channel
                for i in range(8):
                    # Extracting channel data
                    channel = int(parts[i * 3])   # Every 3rd element starting from 0
                    current_value = int(parts[i * 3 + 1])  # Every 3rd element starting from 1
                    
                    # Updating the GUI elements
                    lights[channel].itemconfig(adc_texts[channel], text=f"Channel {channel}: ADC {current_value}")
                    
                    if not(channel < 0 or channel > 7): 
                        # Update the ADC text on canvas
                        lights[channel].itemconfig(adc_texts[channel], text=f"ADC: {current_value}")

                        # Establish baseline in the first second
                        if current_time - start_time <= INITIAL_PERIOD:
                            if current_value != 0:
                                initial_readings[channel].append(current_value)
                                baseline[channel] = sum(initial_readings[channel]) / len(initial_readings[channel])
                        else:
                            # Change light color based on baseline and ADC bound
                            color = calculate_color(current_value, baseline[channel])
                            lights[channel].config(bg=color)

        except UnicodeDecodeError:
            print("Received non-UTF-8 data.")
        except ValueError:
            print("Invalid data received.")

    root.after(10, check_serial)

# Start checking serial data
check_serial()

# Start the GUI event loop
root.mainloop()
