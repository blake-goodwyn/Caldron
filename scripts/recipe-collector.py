## main.py

# Import the necessary libraries
from threads import *
from url_aggregator import createURLFile, run_url_aggregate, urlThreshold
from recipe_scraper import createRecipesFile, run_recipe_scrape
import random
import os
from genai_tools import descriptor_generate

##### Multi-Threaded Approach to Asynchronous/Simulatanous URL Scraping & Aggregation #####
# Define recipe keywords
async def recipe_collector(core_search_term, folder, urlThreshold):

    blacklist = ["reddit.com", "facebook.com", "instagram.com", "pinterest.com", "tiktok.com"]

    descriptors = descriptor_generate(core_search_term)
    print("Selected Descriptors: ", descriptors)
    random.shuffle(descriptors)

    # Create Data Files
    url_file = createURLFile(core_search_term, folder)
    recipe_file = createRecipesFile(core_search_term, folder)

    # Thread 1: URL Aggregation
    thread1 = threading.Thread(target=run_url_aggregate, args=(url_file, core_search_term, urlThreshold, blacklist, descriptors, exception_event, url_queue,))
    thread1.start()

    # Thread 2: Recipe Scraping & Cleaning
    thread2 = threading.Thread(target=run_recipe_scrape, args=(recipe_file, exception_event, url_queue,))
    thread2.start()

    # Wait for queues to be processed
    await url_queue.join()
    await recipe_scraping_queue.join()

    thread1.join()
    thread2.join()

search_terms = ["pizza"]
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
    asyncio.run(recipe_collector(core_search_terms[i], folder, urlThreshold))