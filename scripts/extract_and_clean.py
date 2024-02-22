print("Loading Extract-and-Clean...", end='')
import re
import pandas as pd
from collections import Counter
import ast
import spacy
from threads import *
from whitespace_correction import WhitespaceCorrector

nlp = spacy.load('en_core_web_sm')
cor = WhitespaceCorrector.from_pretrained()
quantity_units = {'tsp', 'teaspoon', 'teaspoons', 'tbsp', 'tablespoon', 'tablespoons', 'cup', 'cups', 'c', 'ml', 'milliliter', 'milliliters', 'liter', 'liters', 'l', 'gram', 'grams', 'g', 'kilogram', 'kilograms', 'kg', 'oz', 'ounce', 'lb', 'pound'}
non_ingredient_keywords = {'ripe', 'softened', 'room', 'temperature', 'ground', 'fresh', 'dried', 'chopped', 'sliced', 'of', 'at', 'with', 'for', 'and'}
ingredient_counter = Counter()

def extract_ingredient(phrase):
    
    # Tokenize the phrase using spaCy
    phrase = cor.correct_text(phrase)
    print(phrase)
    doc = nlp(phrase)
    
    # Initialize variables
    main_ingredient = ''
    quantity = ''
    
    # Iterate over the tokens in the phrase
    for token in doc:
        # Check if the token is a quantity unit
        if token.text.lower() in quantity_units:
            quantity = token.text
        # Check if the token is an ingredient keyword
        elif token.text.lower() not in non_ingredient_keywords and not token.text.isdigit() and not '/' in token.text and not token.text.endswith(' ') and not re.search(r',', token.text):
            main_ingredient += token.text + ' '
    
    # Remove trailing whitespace
    main_ingredient = main_ingredient.strip().lower()
    
    return main_ingredient

def process_ingredient_list(ingredients_list):
    processed_ingredients = []
    print (ingredients_list)

    for ingredient in ingredients_list:
        
        main_ingredient = extract_ingredient(ingredient)
        processed_ingredients.append(main_ingredient)

    print(processed_ingredients)
    return processed_ingredients

def clean_and_extract_ingredients(ingredient_str):
    # Splitting the ingredient string into a list (assuming comma-separated)
    ingredient_list = ast.literal_eval(ingredient_str)
    return process_ingredient_list(ingredient_list)

# Analyzing ingredient frequency
def update_ingredient_counter(ingredient_list, ingredient_counter):
    cleaned = clean_and_extract_ingredients(ingredient_list)
    ingredient_counter.update(cleaned)

def recipe_clean(event):
    while not event.is_set():
        try:
            recipe = recipe_scraping_queue.get()
            try:
                #print(recipe[3])
                print(ast.literal_eval(recipe[3]))
                #print()
                #x = process_ingredient_list(ingredients)

            except Exception as e:
                print(e)

            recipe_scraping_queue.task_done()
        except Exception as e:
            exception_event.set()
    
    print("-- RECIPE CLEAN THREAD EXITING --")

print("COMPLETE!")