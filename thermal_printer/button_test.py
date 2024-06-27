import RPi.GPIO as GPIO
import time

def button_callback(channel):
    print("Button was pressed!")

def setup_gpio():
    try:
        # Use BCM pin numbering
        GPIO.setmode(GPIO.BCM)
        print("Set to BCM mode")

        # Set up GPIO24 as an input with an internal pull-up resistor
        button_pin = 24
        GPIO.setup(button_pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
        print("GPIO set-up complete.")

        # Set up an event detection on the button pin for a falling edge
        GPIO.add_event_detect(button_pin, GPIO.FALLING, callback=button_callback, bouncetime=300)
    except RuntimeError as e:
        print(f"RuntimeError: {e}")
        GPIO.cleanup()
        raise

def main():
    try:
        setup_gpio()
        # Keep the script running
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Exiting program")
    finally:
        GPIO.cleanup()

if __name__ == "__main__":
    main()
