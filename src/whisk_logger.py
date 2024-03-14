import serial
import time
import os
from datetime import datetime

# Set up the serial connection (the COM port may differ on your computer)
ser = serial.Serial('COM4', 9600, timeout=1) # Replace 'COM4' with your Arduino's COM port
time.sleep(2) # Wait for the connection to establish

# Create a timestamp for the filename
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"accelerometer_data_{timestamp}.txt"

# Create a directory for storing data if it doesn't exist
output_directory = os.path.join(os.getcwd(), 'whisk_logger')
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Open a file with the timestamp in the filename for writing
with open(os.path.join(output_directory, filename), 'w') as file:
    while True:
        if ser.in_waiting:
            line = ser.readline().decode('utf-8').rstrip()
            print(line) # Print to console (optional)
            file.write(line + '\n') # Write to file
