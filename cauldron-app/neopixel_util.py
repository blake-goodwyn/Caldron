import neopixel
import board
import time

# Neopixel setup
pixel_pin = board.D18
num_pixels = 24
ORDER = neopixel.RGB
pixels = neopixel.NeoPixel(pixel_pin, num_pixels, brightness=0.2, auto_write=False, pixel_order=ORDER)

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