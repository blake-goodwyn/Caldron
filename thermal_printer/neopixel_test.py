import time
import board
import neopixel

# Configure the setup
LED_COUNT = 24      # Number of LEDs in the NeoPixel ring
LED_PIN = board.D18 # GPIO 18 (PWM)

# Create the NeoPixel object
pixels = neopixel.NeoPixel(LED_PIN, LED_COUNT, brightness=0.2, auto_write=False)


def theater_chase(color, wait_ms=50, iterations=10):
    """Movie theater light style chaser animation."""
    for j in range(iterations):
        for q in range(3):
            for i in range(0, LED_COUNT, 3):
                pixels[i + q] = color
            pixels.show()
            time.sleep(wait_ms / 1000.0)
            for i in range(0, LED_COUNT, 3):
                pixels[i + q] = (0, 0, 0)

def wheel(pos):
    """Generate rainbow colors across 0-255 positions."""
    if pos < 0 or pos > 255:
        return (0, 0, 0)
    if pos < 85:
        return (255 - pos * 3, pos * 3, 0)
    elif pos < 170:
        pos -= 85
        return (0, 255 - pos * 3, pos * 3)
    else:
        pos -= 170
        return (pos * 3, 0, 255 - pos * 3)

def rainbow_cycle(wait_ms=20, iterations=5):
    """Draw rainbow that uniformly distributes itself across all pixels."""
    for j in range(256 * iterations):
        for i in range(LED_COUNT):
            pixel_index = (i * 256 // LED_COUNT) + j
            pixels[i] = wheel(pixel_index & 255)
        pixels.show()
        time.sleep(wait_ms / 1000.0)

if __name__ == "__main__":
    try:
        pixels.fill((0,0,0))
        while True:
            pixels[5] = (255,0,0)
            pixels.show()
    except KeyboardInterrupt:
        pixels.fill((0, 0, 0))
        pixels.show()
