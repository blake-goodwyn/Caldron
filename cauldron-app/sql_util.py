import re

def find_SQL_prompt(text):
    # Regex pattern to match content enclosed in %%..%%
    pattern = r"%%(.*?)%%"
    
    # Find all matches of the pattern
    matches = re.findall(pattern, text)
    
    # Remove the matched content from the text
    text_without_matches = re.sub(pattern, '', text)
    
    # Return both matches and the text without matches
    return matches, text_without_matches