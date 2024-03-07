## main.py

# Import the necessary libraries
from threads import *
import logging
from url_aggregator import createURLFile, url_aggregate, urlThreshold
from recipe_scraper import createRecipesFile, recipe_scrape, add_quotation_marks
import random
from datetime import datetime
from genai_tools import *

##### Multi-Threaded Approach to Asynchronous/Simulatanous URL Scraping & Aggregation #####
# Define recipe keywords
def recipe_collector(core_search_term, folder, urlThreshold):

    blacklist = ["reddit.com", "facebook.com", "instagram.com", "pinterest.com", "tiktok.com"]

    descriptors = descriptor_generate(core_search_term)
    descriptors = eval(re.sub(r'[\r\n]+', ' ', descriptors.lower().strip()))
    print("Selected Descriptors: ", descriptors)
    random.shuffle(descriptors)

    # Create Data Files
    url_file = createURLFile(core_search_term, folder)
    recipe_file = createRecipesFile(core_search_term, folder)

    # Thread 1: URL Aggregation
    print("Starting URL Aggregation")
    thread1 = threading.Thread(target=url_aggregate, args=(url_file, core_search_term, urlThreshold, blacklist, descriptors, exception_event,))
    thread1.start()

    # Thread 2: Recipe Scraping & Cleaning
    print("Starting Recipe Scraping")
    thread2 = threading.Thread(target=recipe_scrape, args=(recipe_file, exception_event,))
    thread2.start()

    # Wait for queues to be processed
    url_queue.join()
    recipe_scraping_queue.join()

search_terms = ["bread"] #["muffin"]
data_path = "C:/Users/blake/Documents/GitHub/ebakery/data"
for folder_name in search_terms:
    folder_path = os.path.join(data_path, folder_name.replace(" ", "-"))
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder created: {folder_path}")
    else:
        print(f"Folder already exists: {folder_path}")

core_search_terms = [s + ' recipe' for s in search_terms]

for i in range(0,len(core_search_terms)):
    folder = os.path.join(data_path, search_terms[i].replace(" ", "-"))
    print(folder)
    recipe_collector(core_search_terms[i], folder, urlThreshold)