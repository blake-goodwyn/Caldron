print("Loading Extract-and-Clean...", end='')
from collections import Counter
from threads import *
from genai_tools import *
import re

ingredient_counter = Counter()

def add_quotation_marks(s):
    assert type(s) == str, "Input must be a string"
    # Check if the string is already quoted
    if not (s.startswith('"') and s.endswith('"')) and not (s.startswith("'") and s.endswith("'")):
        # Add quotation marks
        s = '"' + s + '"'
    return s

def ing_clean(string):
    standardized_ingredients = standardize_ingredients(string)
    return add_quotation_marks(re.sub(r'[\r\n]+', ' ', standardized_ingredients.lower().strip().replace("none","None")))

def instr_clean(string):
    standardized_instructions = standardize_instructions(string)
    return add_quotation_marks(re.sub(r'[\r\n]+', ' ', standardized_instructions.lower().strip()))

# Analyzing ingredient frequency
def update_ingredient_counter(ingredient_list, ingredient_counter):
    cleaned = ing_clean(ingredient_list)
    return increment_counter(cleaned, ingredient_counter)

def increment_counter(processed, ingredient_counter, threshold=10):
    for i in eval(processed.strip()):
        ingredient_counter[i[0]] += 1
    print("Most Common Ingredients:")
    for i in ingredient_counter.most_common(threshold):
        print(i)

    return processed

def recipe_clean(event):
    while not event.is_set():
        try:
            recipe = recipe_scraping_queue.get()
            try:
                update_ingredient_counter(recipe[3], ingredient_counter)

            except Exception as e:
                print(e)

            recipe_scraping_queue.task_done()
        except Exception as e:
            exception_event.set()
    
    print("-- RECIPE CLEAN THREAD EXITING --")

print("COMPLETE!")
