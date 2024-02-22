import queue
import threading

# Threading Objects
url_queue = queue.Queue()
recipe_scraping_queue = queue.Queue()
recipe_cleaning_queue = queue.Queue()
exception_event = threading.Event()