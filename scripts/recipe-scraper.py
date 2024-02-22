# Importing required libraries
import requests
from bs4 import BeautifulSoup, Tag
import os
import csv
import time
import uuid
from datetime import datetime
from urllib.parse import urlparse
import mimetypes
import threading

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
    return [li.get_text(strip=True) for li in ul_or_ol.find_all("li") if li.text]

### TODO: Identified Information Factors ###

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

infoHeaders = ['ID','URL', 'Recipe Name', 'Ingredients', 'Instructions']

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

        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Check that the request was successful
        soup = BeautifulSoup(response.content, 'html.parser')

        out = {
            'name': "",
            'ingredients': [],
            'instructions': [],
            'images': []
        }

        # Assuming images are in <img> tags with a specific class
        out['images'] = list(set(img['src'] for img in soup.find_all('img') 
                    if 'src' in img.attrs 
                    and img['src'].endswith(('.jpg', '.jpeg', '.png'))
                    and 'logo' not in img['src']
                    and 'placeholder' not in img['src']))

        # Grabbing recipe name
        try:
            recipe_name = soup.find('h1').get_text(strip=True)
            out['name'] = recipe_name
        except:
            pass

        # Search for the ingredients and instructions
        ingredients, instructions = [], []
        for header in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
            if 'ingredient' in header.get_text(strip=True).lower():
                ingredients_list = find_following_list(header)
                if ingredients_list:
                    ingredients = get_text_list(ingredients_list)
                    out['ingredients'] = ingredients

            elif 'instruction' in header.get_text(strip=True).lower() or 'direction' in header.get_text(strip=True).lower():
                instructions_list = find_following_list(header)
                if instructions_list:
                    instructions = get_text_list(instructions_list)
                    out['instructions'] = instructions

        return out
        
    except requests.exceptions.RequestException as e:
        print(f"Error fetching URL {url}: {e}")
        return None
    
def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path

# Function to generate a simple, random ID
def generate_id():
    return str(uuid.uuid4())[:8]  # Generate a UUID and use the first 8 characters

data = 'data/urls-2024-02-20-1712.csv' #urls file
c = 0

# Create a base directory for today's date
date_str = datetime.now().strftime('%Y-%m-%d-%H%M')
base_dir = os.path.join('data/', date_str)
create_directory(base_dir)

try:
    with open(data, newline='', encoding='utf-8') as csvfile:
        url_reader = csv.reader(csvfile)

        #Generate new recipes file
        output_file = ''.join([base_dir, "/", datetime.now().strftime('%Y-%m-%d-%H%M'),'.csv'])
        with open(output_file, mode='w', newline='', encoding='utf-8') as file:
            csv_writer = csv.writer(file)
            csv_writer.writerow(infoHeaders)

        for row in url_reader:
            url = row[0]
            c += 1
            print(c, "|", url)
            res = get_recipe_info(url)
            
            if res is not None:
                try:
                    website_id = generate_id()

                    #filter for empty entries
                    assert res.get('ingredients') != []
                    assert res.get('instructions') != []

                    with open(output_file, mode='a', newline='', encoding='utf-8') as file:
                        csv_writer = csv.writer(file)
                        # Write the processed data to the CSV file
                        csv_writer.writerow([website_id, url, res.get('name'), res.get('ingredients'), res.get('instructions')])

                    #download all images
                    try:
                        download_all_images(res.get('images'),base_dir,website_id)
                    except Exception as e:
                        print(e)
                    
                except AssertionError as e:
                    pass

            time.sleep(0.05)

except Exception as e:
    print("ERROR:", e)