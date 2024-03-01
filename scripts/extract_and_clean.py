print("Loading Extract-and-Clean...", end='')
import pandas as pd
from collections import Counter
import os
from threads import *
from genai_tools import *
from datetime import datetime

ingredient_counter = Counter()


def add_quotation_marks(s):
    assert type(s) == str, "Input must be a string"
    # Check if the string is already quoted
    if not (s.startswith('"') and s.endswith('"')) and not (s.startswith("'") and s.endswith("'")):
        # Add quotation marks
        s = '"' + s + '"'
    return s

def clean(string):
    return add_quotation_marks(re.sub(r'[\r\n]+', ' ', normalize_ingredients(string).lower().strip()))

# Analyzing ingredient frequency
def update_ingredient_counter(ingredient_list, ingredient_counter):
    cleaned = clean(ingredient_list)
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

#Test: determine malformed processed ingredient lists
#file_path = ''.join(['data/processed-banana-bread-recipes-', datetime.now().strftime('%Y-%m-%d-%H%M'),'.csv'])
#ID_file = 'data/malformed-processed-ingredient-IDs.txt'
#if not os.path.exists(ID_file):
#    with open(ID_file, mode='w', newline='', encoding='utf-8') as file:
#        file.write("Malformed Processed Ingredient Lists:\n")

#df['Processed_Ingredients'] = pd.Series([None]*len(df['Ingredients']), index=df.index)
#c = 0
#for i in df['Ingredients']:
#    print("Processing Recipe", c+1, "of", len(df['Ingredients']))
#    try:
#        #add result of update_ingredient_counter to df in selected row
#        df.loc[c, "Processed_Ingredients"] = update_ingredient_counter(i, ingredient_counter)
#    except Exception as e:
#        print(e)
#        print(df.loc[c,'ID'])
#        with open(ID_file, 'a') as file:
#            file.write(str(df.loc[c,'ID']) + '\n')
#    c += 1
#    df.to_csv(file_path, index=False)
