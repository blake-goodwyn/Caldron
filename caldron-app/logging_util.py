import logging
import os
from datetime import datetime

# Configure logger
def setup_logging(log_level=logging.DEBUG, log_dir='logs'):
    """Sets up the logging configuration."""
    # Create the logs directory if it does not exist
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Generate a log file name with a datetime stamp
    log_file = os.path.join(log_dir, f"app_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

    # Create a custom logger
    logger = logging.getLogger('cauldron')

    # Set the log level
    logger.setLevel(log_level)

    # Create handlers
    c_handler = logging.StreamHandler()
    f_handler = logging.FileHandler(log_file)

    # Set level for handlers
    c_handler.setLevel(log_level)
    f_handler.setLevel(log_level)

    # Create formatters and add it to handlers
    c_format = logging.Formatter('%(asctime)s - %(name)-6s - %(levelname)-6s - %(filename)-12s - %(funcName)s - %(message)s')
    f_format = logging.Formatter('%(asctime)s - %(name)-6s - %(levelname)-6s - %(filename)-12s - %(funcName)s - %(message)s')

    c_handler.setFormatter(c_format)
    f_handler.setFormatter(f_format)

    # Add handlers to the logger
    logger.addHandler(c_handler)
    logger.addHandler(f_handler)

    return logger

# Initialize logger
logger = setup_logging()
