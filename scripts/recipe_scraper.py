print("Loading Recipe-Scraper...", end='')
# Importing required libraries
import requests
from bs4 import BeautifulSoup, Tag
import os
import csv
import uuid
from datetime import datetime
import mimetypes
import threading
from util import create_directory
from threads import *
from extract_and_clean import ing_clean
import recipe_scrapers
import asyncio

### Identified Information Vectors ###

## Core Info
# Name of the Recipe: Essential for identifying and referencing the dish.
# Ingredients: Critical for understanding the components and quantities needed.
# Instructions: Vital for the step-by-step process of preparing the dish.

## Secondary Info
# Images: Finalized images of the dish
# Prep and Cook Time: Critical for planning and understanding the time commitment required.

# Tertiary Info
# User Ratings and Reviews: Offers valuable insights into the popularity and success of the recipe among other cooks.
# Allergen Information: Crucial for those with food allergies or intolerances.
# Dietary Labels: Essential for individuals following specific dietary guidelines (e.g., vegan, gluten-free).

def find_next_tag(tag, target_tags):
    while tag is not None:
        # Move to the next element at the same level
        tag = tag.find_next_sibling()
        if tag and isinstance(tag, Tag):
            if tag.name in target_tags:
                return tag
            # If it's a container, search within it
            found = tag.find(target_tags, recursive=True)
            if found:
                return found
    return None

def find_following_list(tag):
    return find_next_tag(tag, ['ul', 'ol'])

def get_text_list(ul_or_ol):
    """ Extract text from a list of li tags """
    return [li.get_text(strip=True).replace("\n","") for li in ul_or_ol.find_all("li") if li.text]

infoHeaders = ['ID','URL', 'Recipe Name', 'Ingredients', 'Instructions', 'Processed Ingredients']

def download_image(image_url, folder_name, image_name):
    try:
        # Get the content type of the image from the URL
        response = requests.head(image_url)
        content_type = response.headers.get('content-type')
        extension = mimetypes.guess_extension(content_type)

        # Check if the image name already has the proper extension; if not, append it
        if extension and not image_name.lower().endswith(extension):
            image_name += extension

        # Download the image
        response = requests.get(image_url, timeout=10)
        if response.status_code == 200:
            with open(os.path.join(folder_name, image_name), 'wb') as file:
                file.write(response.content)

    except Exception as e:
        print(f"Error downloading {image_url}: {e}")

def download_all_images(image_urls, base_dir, website_id):
    threads = []
    images_dir = ''.join([base_dir,'/', website_id])
    create_directory(images_dir)

    for i, url in enumerate(image_urls):
        image_name = f'{website_id}_{i}'  # Or use a more sophisticated naming scheme
        thread = threading.Thread(target=download_image, args=(url, images_dir, image_name))
        threads.append(thread)
        thread.start()

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

def get_recipe_info(url):
    
    out = {
            'name': "",
            'ingredients': [],
            'instructions': []
        }
    
    try:
        scraper = recipe_scrapers.scrape_me(url, wild_mode=True)
        out['name'] = scraper.title()
        out['ingredients'] = [i.replace("\n", "") for i in scraper.ingredients()]
        out['instructions'] = [i.replace("\n", "") for i in scraper.instructions_list()]
        #print("recipe_scraper: ", out['name'])
        return out

    except:
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3',
                'Accept': '*/*',  # This accepts any content type
            }

            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()  # Check that the request was successful
            soup = BeautifulSoup(response.content, 'html.parser')

            # Grabbing recipe name
            try:
                recipe_name = soup.find('h1').get_text(strip=True)
                out['name'] = recipe_name
            except Exception as e:
                print(e)

            # Search for the ingredients and instructions
            ingredients, instructions = [], []
            for header in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
                if 'ingredient' in header.get_text(strip=True).lower():
                    ingredients_list = find_following_list(header)
                    if ingredients_list:
                        ingredients = get_text_list(ingredients_list)
                        out['ingredients'] = ingredients

                elif 'instruction' in header.get_text(strip=True).lower() or 'direction' in header.get_text(strip=True).lower() or 'process' in header.get_text(strip=True).lower():
                    instructions_list = find_following_list(header)
                    if instructions_list:
                        instructions = get_text_list(instructions_list)
                        out['instructions'] = instructions

            # Add list items that have 'ingredient' in their class attribute
            for li in soup.find_all('li', class_='ingredient'):
                if li.get_text(strip=True) not in ingredients:
                    ingredients.append(li.get_text(strip=True).replace("\n", ""))

            #print("Default: ", out['name'])
            return out
            
        except Exception as e:
            print(f"Error fetching URL {url}: {e}")
            return None
    
# Function to generate a simple, random ID
def generate_id():
    return str(uuid.uuid4())[:8]  # Generate a UUID and use the first 8 characters

async def process(url, file_path):
    print("Processing: ", url)
    res = await asyncio.to_thread(get_recipe_info, url)
    if res is not None:
        try:
            website_id = generate_id()

            #filter for empty entries
            assert res.get('ingredients') != []
            assert res.get('instructions') != []
            assert res.get('name') != ""

            processed_ingredients = await asyncio.to_thread(ing_clean, str(res.get('ingredients')))
            print("Processed: ", res.get('name'))

            await asyncio.to_thread(write_to_csv, file_path, website_id, url, res, processed_ingredients)
                
        except AssertionError as e:
            print(e)
        except Exception as e:
            print(e)
    else:
        print("Error fetching URL: ", url)

def write_to_csv(file_path, website_id, url, res, processed_ingredients):
    with open(file_path, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        row_data = [
            website_id,
            url,
            res.get('name'),
            ', '.join(res.get('ingredients')) if isinstance(res.get('ingredients'), list) else res.get('ingredients'),
            ', '.join(res.get('instructions')) if isinstance(res.get('instructions'), list) else res.get('instructions'),
            ', '.join(eval(processed_ingredients)) if isinstance(eval(processed_ingredients), list) else eval(processed_ingredients)
        ]
        writer.writerow(row_data)

async def recipe_scrape(file_path, exception_event, url_queue):
    while not exception_event.is_set() or not url_queue.empty():
        try:
            # Wait for at least one URL to be available or timeout
            try:
                await asyncio.wait_for(url_queue.get(), timeout=10)
                url_queue.task_done()  # Mark the first URL as done
            except asyncio.TimeoutError:
                continue  # No URL was found, check the loop condition again

            # Gather URLs from the queue without blocking
            urls = []
            while not url_queue.empty():
                urls.append(url_queue.get_nowait())
                url_queue.task_done()

            # Create asynchronous process tasks for each URL
            tasks = [process(url, file_path) for url in urls]

            # Wait for all tasks to complete
            await asyncio.gather(*tasks)

        except Exception as e:
            print(e)
    
    print("-- RECIPE SCRAPE THREAD EXITING --")

def run_recipe_scrape(recipe_file, exception_event, url_queue):
    print("Starting Recipe Scraping")
    asyncio.set_event_loop(asyncio.new_event_loop())
    loop = asyncio.get_event_loop()
    loop.run_until_complete(recipe_scrape(recipe_file, exception_event, url_queue))
    loop.close()

# Creates a CSV file to store recipe scraping
def createRecipesFile(search_term, folder_path):
    file_path = ''.join([folder_path,'/processed-', search_term.strip().replace(" ","-"), '-', datetime.now().strftime('%Y-%m-%d-%H%M'),'.csv'])
    with open(file_path, mode='w', newline='', encoding='utf-8') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow(infoHeaders)
    return file_path

def createMalformedFile():
    ID_file = 'data/malformed-processed-ingredient-IDs.txt'
    if not os.path.exists(ID_file):
        with open(ID_file, mode='w', newline='', encoding='utf-8') as file:
            file.write("Malformed Processed Ingredient Lists:\n")
            file.close()
    return ID_file

#ID_file = createMalformedFile()
print("COMPLETE!")