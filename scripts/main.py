## main.py

# Import the necessary libraries
from threads import *
from url_aggregator import createURLFile, url_aggregate, urlThreshold
from recipe_scraper import createRecipesFile, recipe_scrape
from extract_and_clean import recipe_clean
import csv
import random

ACTIVE = True  # Set to True to run the URL Aggregation and Recipe Scraping threads

##### Multi-Threaded Approach to Asynchronous/Simulatanous URL Scraping & Aggregation #####
# Define recipe keywords
core_search_term = "banana bread recipe"
blacklist = ["reddit.com", "facebook.com", "instagram.com", "pinterest.com"]
descriptors = [
    "classic",
    "moist",
    "vegan",
    "gluten-free",
    "nutty",
    "chocolate chip",
    "low sugar",
    "whole wheat",
    "healthy",
    "cinnamon",
    "blueberry",
    "pumpkin spice",
    "crunchy topping",
    "dairy-free",
    "sugar-free",
    "high protein",
    "low fat",
    "walnut",
    "pecan",
    "with oats",
    "coconut",
    "almond flour",
    "spiced",
    "with yogurt",
    "with sour cream",
    "maple sweetened",
    "honey",
    "apple cinnamon",
    "berry",
    "raisin",
    "with cream cheese",
    "peanut butter",
    "double chocolate",
    "dark chocolate",
    "with orange zest",
    "with streusel topping",
    "olive oil",
    "buttermilk",
    "with molasses",
    "with vanilla",
    "with dates",
    "with figs",
    "no-bake",
    "with zucchini",
    "with pineapple",
    "with applesauce",
    "caramel",
    "with espresso",
    "with ginger",
    "with carrot"
]
random.shuffle(descriptors)

if ACTIVE:
    # Create Data Files
    url_file = createURLFile()
    recipe_file = createRecipesFile()

    # Thread 1: URL Aggregation
    print("Starting URL Aggregation")
    thread1 = threading.Thread(target=url_aggregate, args=(url_file, core_search_term, urlThreshold, blacklist, descriptors, exception_event,))
    thread1.start()

    # Thread 2: Recipe Scraping
    print("Starting Recipe Scraping")
    thread2 = threading.Thread(target=recipe_scrape, args=(recipe_file, exception_event,))
    thread2.start()
else:
    recipe_file = "ebakery/data/recipes-2024-02-22-1551.csv"
    # Read through recipe_file CSV and create entries in the recipe_scraping_queue
    with open(recipe_file, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header row
        for row in reader:
            recipe_scraping_queue.put(row)

# Thread 3: Recipe Cleaning
print("Starting Recipe Cleaning")
thread3 = threading.Thread(target=recipe_clean, args=(exception_event,))
thread3.start()

# Wait for queues to be processed
url_queue.join()
recipe_scraping_queue.join()
recipe_cleaning_queue.join()

# TODO: Thread 4: Recipe Analysis
## When Recipe Cleaning adds a recipe to the CSV file, analyze the recipe information by continuously printing and updating a printout of the most common ingredients