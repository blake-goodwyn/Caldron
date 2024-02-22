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
removals = {'cups', 'cup'}
ingredient_counter = Counter()

def preprocess_phrase(phrase):
    # Insert space before and after digits and special characters
    phrase = cor.correct_text(phrase).lower()
    phrase = re.sub(r"([0-9]+)", r" \1 ", phrase)
    phrase = re.sub(r"([^\w\s-])", r" \1 ", phrase)
    phrase = re.sub(r"(\d+\s?⁄\s?\d+)", r" \1 ", phrase)
    return phrase

def split_quantity_units(phrase):
    words = []
    # Sort the quantity units by length in descending order
    sorted_units = sorted(removals, key=len, reverse=True)
    for word in phrase.split():
        for unit in sorted_units:
            if word.lower().startswith(unit):
                # Remove the longest matching quantity unit
                word = word[len(unit):]
                break
        words.append(word)
    return ' '.join(words)

def extract_ingredient(phrase):
    #print("RAW: ", phrase)
    phrase = preprocess_phrase(phrase)
    phrase = split_quantity_units(phrase)
    #print("INPUT: ", phrase)
    doc = nlp(phrase)
    main_ingredient = ''
    ingredient_started = False

    for token in doc:
        if token.text.lower() in quantity_units or token.pos_ == 'NUM':
            ingredient_started = True
        elif ingredient_started and (token.pos_ in ['NOUN', 'ADJ']) and (token.text.lower() not in non_ingredient_keywords):
            if token.dep_ == 'compound' or (token.head.pos_ == 'NOUN' and token.head.dep_ != 'appos'):
                main_ingredient += token.text + ' '
            elif token.pos_ == 'NOUN' and not main_ingredient:
                main_ingredient += token.text + ' '
                break  # Stop after finding the first main ingredient

    main_ingredient = main_ingredient.strip().lower()
    #print("OUTPUT: ", main_ingredient)
    return main_ingredient

def process_ingredient_list(ingredients_list):
    processed_ingredients = []

    for ingredient in ingredients_list:
        main_ingredient = extract_ingredient(ingredient)
        processed_ingredients.append(main_ingredient)

    return processed_ingredients

def clean_and_extract_ingredients(ingredient_str):
    # Splitting the ingredient string into a list (assuming comma-separated)
    ingredient_list = ast.literal_eval(ingredient_str)
    return process_ingredient_list(ingredient_list)

# Analyzing ingredient frequency
def update_ingredient_counter(ingredient_list, ingredient_counter):
    cleaned = clean_and_extract_ingredients(ingredient_list)
    ingredient_counter.update(cleaned)
    os.system('clear') if os.name == 'posix' else os.system('cls')
    print("Most Common Ingredients:")
    print(ingredient_counter.most_common(20))

def recipe_clean(event):
    while not event.is_set():
        try:
            recipe = recipe_scraping_queue.get()
            try:
                ingredients = ast.literal_eval(recipe[3])
                #print("THREAD: ", ingredients)
                x = process_ingredient_list(ingredients)

            except Exception as e:
                print(e)

            recipe_scraping_queue.task_done()
        except Exception as e:
            exception_event.set()
    
    print("-- RECIPE CLEAN THREAD EXITING --")

print("COMPLETE!")