print("Loading Recipe-Scraper...", end='')
# Importing required libraries
import requests
from bs4 import BeautifulSoup, NavigableString, Tag
import os
import csv
import time
import uuid
from datetime import datetime
import mimetypes
import threading
from util import create_directory
from threads import *
from extract_and_clean import clean, update_ingredient_counter, ingredient_counter, add_quotation_marks

def get_containers(start_header, end_header, soup):
    # Collect all containers between the start and end headers recursively
    containers = []

    def recurse_through_siblings(element):
        nonlocal containers
        while element and element != end_header:
            if element.name in ['ul', 'ol']:
                containers.append(element)
            elif element.find(['ul', 'ol']):
                containers.extend(element.find_all(['ul', 'ol'], recursive=False))
            # Recurse through child elements
            for child in element.find_all(recursive=False):
                recurse_through_siblings(child)
            element = element.find_next_sibling()

    recurse_through_siblings(start_header.find_next_sibling())
    return containers

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

def get_text_list(elements):
    """ Extract text from a list of li tags """
    return [element.get_text(strip=True) for element in elements if element.text]

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
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3',
            'Accept': '*/*',  # This accepts any content type
        }

        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()  # Check that the request was successful
        soup = BeautifulSoup(response.content, 'html.parser')

        out = {
            'name': "",
            'ingredients': [],
            'instructions': [],
            'images': []
        }

        # Grabbing recipe name
        try:
            recipe_name = soup.find('h1').get_text(strip=True)
            out['name'] = recipe_name
        except:
            pass

        # Find positions of Ingredients and Instructions headers
        ingredients_header, instructions_header = None, None

        # Locate Ingredients and Instructions Headers
        for header in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
            header_text = header.get_text(strip=True).lower()
            if 'ingredient' in header_text and not ingredients_header:
                ingredients_header = header
            elif ('instruction' in header_text or 'direction' in header_text) and not instructions_header:
                instructions_header = header

        # Extract Ingredients from Containers
        if ingredients_header and instructions_header:
            ingredients_containers = get_containers(ingredients_header, instructions_header, soup)
            for container in ingredients_containers:
                out['ingredients'].extend([li.get_text(strip=True) for li in container.find_all("li")])

        # Search for the ingredients and instructions
        for header in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
            if 'instruction' in header.get_text(strip=True).lower() or 'direction' in header.get_text(strip=True).lower():
                instructions_list = find_following_list(header)
                if instructions_list:
                    instructions = get_text_list(instructions_list)
                    out['instructions'] = instructions

        return out
        
    except requests.exceptions.RequestException as e:
        print(f"Error fetching URL {url}: {e}")
        return None
    
# Function to generate a simple, random ID
def generate_id():
    return str(uuid.uuid4())[:8]  # Generate a UUID and use the first 8 characters

def process(url, file_path):
    res = get_recipe_info(url)
    print("Processing: ", url)
    if res is not None:
        try:
            website_id = generate_id()

            #filter for empty entries
            assert res.get('ingredients') != []
            assert res.get('instructions') != []
            assert res.get('name') != ""

            processed_ingredients = clean(str(res.get('ingredients')))
            print("Processed: ", res.get('name'))
            print("URL Queue Size: ", url_queue.qsize())

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
                
        except AssertionError as e:
            print(e)
        except Exception as e:
            print(e)

def recipe_scrape(file_path, exception_event):
    while not exception_event.is_set() or not url_queue.empty():
        try:
            url = url_queue.get()
            process(url, file_path)
            url_queue.task_done()
        except Exception as e:
            pass
            #print(e)
    
    print("-- RECIPE SCRAPE THREAD EXITING --")

# Creates a CSV file to store recipe scraping
def createRecipesFile(search_term, folder_path):
    file_path = ''.join([folder_path,'/processed-', search_term.strip().replace(" ","-"), '-', datetime.now().strftime('%Y-%m-%d-%H%M'),'.csv'])
    with open(file_path, mode='w', newline='', encoding='utf-8') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow(['ID','URL', 'Recipe Name', 'Ingredients', 'Instructions', 'Processed Ingredients'])
    return file_path

def createMalformedFile():
    ID_file = 'data/malformed-processed-ingredient-IDs.txt'
    if not os.path.exists(ID_file):
        with open(ID_file, mode='w', newline='', encoding='utf-8') as file:
            file.write("Malformed Processed Ingredient Lists:\n")
    return ID_file

ID_file = createMalformedFile()
print("COMPLETE!")