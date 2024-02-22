from googleapiclient.discovery import build
import pprint
import pandas as pd
from datetime import datetime
import random
import time

def google_search(search_term, api_key, cse_id, start_num, **kwargs):
    service = build("customsearch", "v1", developerKey=api_key)
    res = service.cse().list(q=search_term, cx=cse_id, start=start_num, **kwargs).execute()
    return res

# Replace with your API key and CSE ID
api_key = "AIzaSyBqffLzRrNKUQX-nZiU8NEp1ocB1P9MeHI"
cse_id = "6373f179be4354964"

desired_number_of_urls = 1000  # total number of URLs you want
core_search_term = "banana bread recipe"
banana_bread_descriptors = [
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

random.shuffle(banana_bread_descriptors)
urls = set()

blacklisted_domains = ["reddit.com", "facebook.com", "instagram.com", "pinterest.com"]
check1 = time.time()
try:
    for d in banana_bread_descriptors:
        term = ''.join([core_search_term, " ", d])
        for start_num in range(1, 91, 10):  # increment by 10 as API allows max 10 results at a time
            check2 = time.time()
            if (check2-check1) < 2:
                time.sleep(2-(check2-check1)) #rate limit safeguard
            search_results = google_search(term, api_key, cse_id, start_num)
            for result in search_results.get('items', []):
                url = result.get('link')
                if not any(domain in url for domain in blacklisted_domains):
                    urls.add(url)
                    print(len(urls), "|", url)

                if len(urls) >= desired_number_of_urls:
                    break

            if len(urls) >= desired_number_of_urls:
                break
        if len(urls) >= desired_number_of_urls:
                break
except:
    print("BREAK!")

urls = list(urls)
print("# of URLs: ", len(urls))

# Assuming 'data' is your aggregated data
if urls is not []:
  df = pd.DataFrame(urls)

# Save to a CSV file
file_path = ''.join(['data/urls-', datetime.now().strftime('%Y-%m-%d-%H%M'),'.csv'])
df.to_csv(file_path, index=False)