from cauldron_app import CaldronApp
from custom_print import wrapper
from logging_util import logger

def app_request(transcription):
    logger.info("Posting audio to Caldron App")
    logger.info(f"Transcription: {transcription}")
    # Post audio to Caldron App
    try:
        app = CaldronApp("gpt-3.5-turbo", verbose=True)
        app.post(transcription)
    except:
        out = "Sorry, I ran into an error. :( This happens sometimes as I'm still learning. Feel free to try again with a different request!"
        logger.error(out)
        print(wrapper.fill(out))