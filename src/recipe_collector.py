## main.py

# Import the necessary libraries
from threads import *
from url_aggregator import createURLFile, run_url_aggregate
from recipe_scraper import createRecipesFile, run_recipe_scrape
import random
import os
from genai_tools import descriptor_generate
from tqdm import tqdm

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

def recipe_collect(search_terms, data_path, urlThreshold):

    assert type(search_terms) == list or type(search_terms) == str, "Search terms must be a list or string"
    if type(search_terms) == str:
        search_terms = [search_terms]

    #Assert that the data path exists
    assert os.path.exists(data_path), "Data path does not exist"
    assert type(urlThreshold) == int, "URL Threshold must be an integer"
    assert urlThreshold > 0, "URL Threshold must be greater than 0"
    
    # Create folders for each search term
    for folder_name in search_terms:
        folder_path = os.path.join(data_path, folder_name.replace(" ", "_"))
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            print(f"Folder created: {folder_path}")
        else:
            print(f"Folder already exists: {folder_path}")

    core_search_terms = [s + ' recipe' for s in search_terms]

    for i in range(0,len(core_search_terms)):
        folder = os.path.join(data_path, search_terms[i].replace(" ", "_"))
        asyncio.run(recipe_collector(core_search_terms[i], folder, urlThreshold))