print("Loading URL-Aggregator...", end='')
from googleapiclient.discovery import build
import pandas as pd
from datetime import datetime
import time
import csv
from threads import *
import uuid

## URL Aggregation Functions ##
api_key = "AIzaSyBqffLzRrNKUQX-nZiU8NEp1ocB1P9MeHI"
cse_id = "6373f179be4354964"

urlThreshold = 3000  # total number of URLs you want
urls = set()

def url_aggregate(file_path, core_search_term, desired_number_of_urls, blacklisted_domains, recipe_descriptors, event):
    """
    Aggregate URLs based on a core search term and desired number of URLs.

    Args:
        file_path (str): The file path to save the URLs.
        core_search_term (str): The core search term to use for URL aggregation.
        desired_number_of_urls (int): The desired number of URLs to aggregate.
        event (threading.Event): An event object to signal the termination of URL aggregation.

    Returns:
        None
    """
    check1 = time.time()
    for d in recipe_descriptors:
        term = f"{core_search_term} {d}"
        for start_num in range(1, 91, 10):  # increment by 10 as API allows max 10 results at a time
            check2 = time.time()
            if (check2 - check1) < 2:
                time.sleep(2)  # rate limit safeguard
            try:
                search_results = google_search(term, api_key, cse_id, start_num)
                for result in search_results.get('items', []):
                    url = result.get('link')
                    if not any(domain in url for domain in blacklisted_domains) and url not in urls:
                        urls.add(url)
                        url_queue.put(url)
                        with open(file_path, mode='a', newline='', encoding='utf-8') as file:
                            csv_writer = csv.writer(file)
                            csv_writer.writerow([url])

                    if len(urls) >= desired_number_of_urls:
                        break
            except Exception as e:
                print(e)        

            if len(urls) >= desired_number_of_urls:
                break
        if len(urls) >= desired_number_of_urls:
            break
    
    print(f"URL Aggregation Complete: {len(urls)} URLs found")
    event.set()

def google_search(search_term, api_key, cse_id, start_num, **kwargs):
    """
    Perform a Google search using the Custom Search JSON API.

    Args:
        search_term (str): The search term to query.
        api_key (str): The API key for accessing the Custom Search JSON API.
        cse_id (str): The Custom Search Engine ID.
        start_num (int): The index of the first search result to return.
        **kwargs: Additional keyword arguments to pass to the search API.

    Returns:
        dict: The search results as a dictionary.

    """
    service = build("customsearch", "v1", developerKey=api_key)
    res = service.cse().list(q=search_term, cx=cse_id, start=start_num, **kwargs).execute()
    return res

def createURLFile(search_term, folder_path):
    """
    Creates a CSV file for storing URLs with a unique name based on the current date and time.
    
    Returns:
        str: The file path of the created CSV file.
    """
    file_path = ''.join([folder_path,'/', search_term.strip().replace(" ","-"), "-urls-", datetime.now().strftime('%Y-%m-%d-%H%M'),'.csv'])
    with open(file_path, mode='w', newline='', encoding='utf-8') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow(["URL"])
    return file_path

def find_next_tag(tag, target_tags):
    """
    Finds the next tag in the HTML document that matches any of the target tags.

    Args:
        tag (Tag): The current tag to start the search from.
        target_tags (list): A list of target tags to search for.

    Returns:
        Tag or None: The next tag that matches any of the target tags, or None if not found.
    """
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
    """
    Finds the next <ul> or <ol> tag following the given tag.

    Parameters:
    tag (Tag): The starting tag to search from.

    Returns:
    Tag: The next <ul> or <ol> tag found, or None if not found.
    """
    return find_next_tag(tag, ['ul', 'ol'])

def get_text_list(ul_or_ol):
    """
    Extracts the text content from the list items within a given unordered or ordered list.

    Args:
        ul_or_ol (BeautifulSoup object): The unordered or ordered list to extract text from.

    Returns:
        list: A list of text content from the list items.

    """
    return [li.get_text(strip=True) for li in ul_or_ol.find_all("li") if li.text]

def generate_id():
    return str(uuid.uuid4())[:8]  # Generate a UUID and use the first 8 characters

print("COMPLETE!")