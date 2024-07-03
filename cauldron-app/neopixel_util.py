import neopixel
import board
import time
import threading
import math

# Neopixel setup
pixel_pin = board.D18
num_pixels = 24
ORDER = neopixel.RGB
pixels = neopixel.NeoPixel(pixel_pin, num_pixels, brightness=0.2, auto_write=False, pixel_order=ORDER)

waiting_color = (255, 178, 0)
thread = None
stop_event = threading.Event()

def gaussian(x, mu, sigma):
    """ Gaussian function for brightness distribution """
    return math.exp(-0.5 * ((x - mu) / sigma) ** 2)

def run_in_thread(target, *args):
    global thread, stop_event
    if thread and thread.is_alive():
        stop_event.set()
        thread.join()
    stop_event.clear()
    thread = threading.Thread(target=target, args=args)
    thread.daemon = True
    thread.start()

def rotate_brightness(loop_duration=2, cycles=5, sigma=1.0):
    """ Rotate brightness around the NeoPixel ring with Gaussian spillover """
    def rotate():
        wait_time = loop_duration / num_pixels  # Calculate the wait time for each step to match the loop duration
        while not stop_event.is_set():
            for cycle in range(cycles):
                for i in range(num_pixels):
                    if stop_event.is_set():
                        clear_ring(internal_call=True)
                        return
                    for j in range(num_pixels):
                        distance = min(abs(j - i), num_pixels - abs(j - i))  # Handle wrap-around
                        brightness = gaussian(distance, 0, sigma)
                        pixels[j] = (
                            int(waiting_color[0] * brightness),
                            int(waiting_color[1] * brightness),
                            int(waiting_color[2] * brightness)
                        )
                    pixels.show()
                    time.sleep(wait_time)
            
        clear_ring(internal_call=True)
        
    run_in_thread(rotate)

def color_wipe(max_brightness=0.2, wait_ms=1500):
    """ Smoothly wipe brightness across the NeoPixel ring """
    
    def wipe():
        steps = int(wait_ms / 100)  # Number of steps to achieve smooth transition
        brightness_step = max_brightness / steps
        while not stop_event.is_set():
            for i in range(steps + 1):
                if stop_event.is_set():
                    clear_ring(internal_call=True)
                    return
                current_brightness = brightness_step * i
                pixels.fill((int(waiting_color[0] * current_brightness), 
                             int(waiting_color[1] * current_brightness), 
                             int(waiting_color[2] * current_brightness)))
                pixels.show()
                time.sleep(wait_ms / (steps * 1000))  # Adjust to keep the same total time
                
            # Decreasing brightness
            for i in range(steps, -1, -1):
                if stop_event.is_set():
                    clear_ring(internal_call=True)
                    return
                current_brightness = brightness_step * i
                pixels.fill((int(waiting_color[0] * current_brightness), 
                             int(waiting_color[1] * current_brightness), 
                             int(waiting_color[2] * current_brightness)))
                pixels.show()
                time.sleep(wait_ms / (steps * 1000))  # Adjust to keep the same total time
                
        clear_ring(internal_call=True)
        
    run_in_thread(wipe)

def clear_ring(internal_call=False):
    stop_event.set()
    if not internal_call and thread and thread.is_alive():
        thread.join()
    pixels.fill((0, 0, 0))  # Turn off all pixels after animation
    pixels.show()

def highlight_section(agent, max_brightness=1.0):
    global thread, stop_event
    
    agent_locs = {
        'Tavily': [20, 21, 22],
        'Frontman': [5, 6, 7],
        'Spinnaret': [10, 11, 12],
        'Sleuth': [15, 16, 17]
    }
    
    def pulse():
        while not stop_event.is_set():
            for brightness in range(0, 101, 5):  # Increase brightness (0 to 100)
                if stop_event.is_set():
                    clear_ring(internal_call=True)
                    break
                current_brightness = brightness / 100.0 * max_brightness
                for i in agent_locs[agent]:
                    pixels[i] = (
                        int(waiting_color[0] * current_brightness),
                        int(waiting_color[1] * current_brightness),
                        int(waiting_color[2] * current_brightness)
                    )
                pixels.show()
                time.sleep(0.01)
            for brightness in range(100, -1, -5):  # Decrease brightness (100 to 0)
                if stop_event.is_set():
                    clear_ring(internal_call=True)
                    break
                current_brightness = brightness / 100.0 * max_brightness
                for i in agent_locs[agent]:
                    pixels[i] = (
                        int(waiting_color[0] * current_brightness),
                        int(waiting_color[1] * current_brightness),
                        int(waiting_color[2] * current_brightness)
                    )
                pixels.show()
                time.sleep(0.01)
            
        clear_ring(internal_call=True)
    
    # Start pulsing in a new thread
    run_in_thread(pulse)
