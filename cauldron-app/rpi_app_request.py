from custom_print import wrapper
from logging_util import logger
from thermal_printer_util import printer

def app_request(app, transcription):
    logger.info("Posting audio to Caldron App")
    logger.info(f"Transcription: {transcription}")
    # Post audio to Caldron App
    try:
        app.clear_pot()
        app.post(transcription)
    except:
        out = "Sorry, I ran into an error. :( This happens sometimes as I'm still learning. Feel free to try again with a different request!"
        logger.error(out)
        printer.print(wrapper.fill(out))
