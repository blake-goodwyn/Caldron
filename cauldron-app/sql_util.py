import re

def find_SQL_prompt(text):
    # Regex pattern to match content enclosed in %%..%%
    pattern = r"%%(.*?)%%"
    
    # Find all matches of the pattern
    matches = re.findall(pattern, text)
    
    # Return both matches and the text without matches
    return matches