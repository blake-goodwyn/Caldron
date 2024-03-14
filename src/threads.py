import threading
import asyncio

# Threading Objects
url_queue = asyncio.Queue()
recipe_scraping_queue = asyncio.Queue()
exception_event = threading.Event()