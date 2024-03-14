import pandas as pd
import os
import matplotlib.pyplot as plt
from collections import Counter
from itertools import combinations
import nltk
import sys
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
import statistics as stat
from dataset_linter import lint_dataset

_DEBUG = False

def display_loading_bar(current, total, bar_length=50):
    """
    Display a loading bar in the console.

    :param current: Current progress (current index).
    :param total: Total count.
    :param bar_length: Length of the loading bar (default 50 characters).
    """
    progress = float(current) / total
    arrow = '#' * int(round(progress * bar_length) - 1) + '>'
    spaces = ' ' * (bar_length - len(arrow))

    sys.stdout.write("\r[{0}] {1}%".format(arrow + spaces, int(round(progress * 100))))
    sys.stdout.flush()

def get_wordnet_pos(word):
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)

def lemmatize_ingredient(phrase):
    
    known_compounds = {'brown sugar', 
                       'chocolate chip', 
                       'cream cheese', 
                       'vanilla extract', 
                       'baking powder', 
                       'baking soda', 
                       'chocolate chips'}

    # Check if the entire phrase is a known compound noun
    if phrase.lower() in known_compounds:
        return phrase

    lemmatizer = WordNetLemmatizer()
    words = word_tokenize(phrase)
    pos_tags = nltk.pos_tag(words)

    # Filter out words that are not nouns
    nouns = [word for word, pos in pos_tags if pos.startswith('NN')]

    # Lemmatize the nouns
    lemmatized_nouns = [lemmatizer.lemmatize(noun, get_wordnet_pos(noun)) for noun in nouns]

    # Return the last noun (usually the main ingredient)
    return lemmatized_nouns[-1] if lemmatized_nouns else ''

# Define a conversion function to standardize measurements to cups
def normalize_to_flour(quantity, unit):
    conversion_factors = {
        'g': 0.00422675,  # Grams to cups (for flour, approximately)
        'tablespoon': 0.0625,  # Tablespoons to cups
        'tablespoons': 0.0625,  # Tablespoons to cups
        'tbsp': 0.0625,  # Tablespoons to cups
        'tsp': 0.0208333,  # Teaspoons to cups
        'teaspoon': 0.0208333,  # Teaspoons to cups
        'teaspoons': 0.0208333,  # Teaspoons to cups
        'cup': 1,  # Cups to cups
        'cups': 1,  # Cups to cups
        'ounce': 0.125,  # Ounces to cups
        'ounces': 0.125,  # Ounces to cups
        'oz': 0.125,  # Ounces to cups
        'milliliter': 0.00422675,
        'milliliters': 0.00422675,
        'ml': 0.00422675,
        'liter': 4.22675,
        'liters': 4.22675,
        'l': 4.22675,
    }
    return quantity * conversion_factors.get(unit, 1)  # Default is 1 if unit is already in cups or unknown

def normalize_ingredients(df):
    # Process each recipe
    c = 1
    for index, row in df.iterrows():
        display_loading_bar(index, len(df))
        try:
            ingredients = eval(row['Processed Ingredients'])
            flour_cups = 0
            normalized_ingredients = []

            # Accumulate the quantity of flour in cups
            for ingredient in ingredients:
                name, quantity, unit = ingredient
                if "flour" in name.lower():
                    flour_cups += normalize_to_flour(quantity, unit)

            # Skip normalization if flour is not found or has zero quantity
            if flour_cups == 0:
                continue

            # Add the total flour quantity to the normalized ingredients
            normalized_ingredients.append(("flour", 1, 'cup'))

            # Normalize other ingredients against the total quantity of flour
            for ingredient in ingredients:
                name, quantity, unit = ingredient
                if "flour" not in name.lower():
                    # Lemmatize ingredient name
                    name = lemmatize_ingredient(name)
                    # Convert ingredient name to its singular form
                    quantity_in_cups = normalize_to_flour(quantity, unit)
                    normalized_quantity = quantity_in_cups / flour_cups
                    normalized_ingredients.append((name, normalized_quantity, 'cups'))

            #Combine ingredients that are now identical after lemmatization
            combined_ingredients = {}
            for name, quantity, unit in normalized_ingredients:
                if name in combined_ingredients:
                    combined_ingredients[name] = (combined_ingredients[name][0] + quantity, unit)
                else:
                    combined_ingredients[name] = (quantity, unit)

            normalized_ingredients = [(name, qty, unit) for name, (qty, unit) in combined_ingredients.items()]

            # Store the normalized ingredients
            df.at[index, 'Normalized Ingredients'] = normalized_ingredients
            
        except Exception as e:
            if _DEBUG:
                print(e)

    return df

def update_co_occurrence(ing_list, co_occurrence):

    assert type(ing_list) == list, "Input must be a list"
    assert type(co_occurrence) == Counter, "Input must be a Counter object"

    # Extract only ingredient names for co-occurrence calculation
    ingredient_names = [ingredient[0] for ingredient in ing_list]

    # Update co-occurrence counts
    for ing1, ing2 in combinations(ingredient_names, 2):
        if ing1 != ing2:
            co_occurrence.update([(ing1, ing2)])

    return co_occurrence

def show_co_occurrence(co_occurrence, names, ing_pos, prevalences):
    # Draw lines based on co-occurrence
    min_co_occurrence = 0.1*max(prevalences)  # Set your threshold

    # Find the maximum co-occurrence count
    max_co_occurrence = max(co_occurrence.values())

    for ingredients, count in co_occurrence.items():
        if count >= min_co_occurrence:
            ing1, ing2 = ingredients
            if ing1 in names and ing2 in names:
                x1, y1 = ing_pos[ing1]
                x2, y2 = ing_pos[ing2]

                # Scale opacity and linewidth based on the maximum co-occurrence
                alpha = count / max_co_occurrence  # Scales from 0 to 1
                linewidth = (count / max_co_occurrence) * 2  # Scales up to 5

                plt.plot([x1, x2], [y1, y2], color='grey', alpha=alpha, linewidth=linewidth)

def visualize(file, recipe_type):
    # Load the data
    df = lint_dataset(pd.read_csv(file))
    df = normalize_ingredients(df)
    #co_occurrence = Counter()
    dataset_count = 0
    total_ing = 0
    failed_ing = 0

    # Process the ingredient data
    ingredient_data = {}
    for index, row in df.iterrows():
        try:
            if _DEBUG:
                print(row['Normalized Ingredients'])
            ingredients = row['Normalized Ingredients']  # Replace with actual column name
            #co_occurrence = update_co_occurrence(ingredients, co_occurrence)
            total_ing += len(ingredients)
            for ingredient in ingredients:
                try:
                    name, quantity, unit = ingredient
                    if name not in ingredient_data:
                        ingredient_data[name] = {'total_quantity': [], 'recipe_count': 0}
                    ingredient_data[name]['total_quantity'].append(quantity)  # Adjusted via Normalized Ingredients
                    ingredient_data[name]['recipe_count'] += 1
                except Exception as e:
                    failed_ing += 1
                    if _DEBUG:
                        print(e)

            dataset_count+=1
        except Exception as e:
            if _DEBUG:
                print(e)

    # Prepare data for plotting
    names = []
    quantities = []
    prevalences = []
    errors = []
    for name, data in sorted(ingredient_data.items(),key=lambda x: x[1]['recipe_count'], reverse=True):
        names.append(name)
        quantities.append(stat.median(data['total_quantity']))
        try:
            errors.append(stat.stdev(data['total_quantity']))
        except:
            errors.append([0,0])
        prevalences.append(data['recipe_count']/dataset_count)

    #Filter for top ingredients
    k=10
    names = names[:k]
    quantities = quantities[:k]
    errors = errors[:k]

    # Create the bar graph
    plt.figure(figsize=(10, 6))
    plt.bar(names, quantities, alpha=0.7)  # Adjust alpha as needed
    #plt.errorbar(names, quantities, yerr=errors, fmt='o', color='black', alpha=0.7)  # Adjust alpha as needed
    plt.ylabel('Relative Quantity (Cups)')
    plt.xticks(rotation=20)
    plt.title(f'Ingredient Quantities in {recipe_type} Recipes (n = {dataset_count})')

    ##For scatter variant
    #show_co_occurrence(co_occurrence, names, ing_pos, prevalences):

    print("\nPercentage of rows readable: ", dataset_count/len(df)*100, "%")
    print("Percentage of ingredients readable: ", (total_ing-failed_ing)/total_ing*100, "%")

    plt.show()


visualize('data/GOOD DATASETS/processed-cookie-recipe-2024-03-04-1918.csv', "Cookie")
#folder = 'data/GOOD DATASETS/'
#folderObj = os.scandir(folder)
#for entry in folderObj:
#    if entry.is_file():
#        file = folder + entry.name
#        visualize(file, entry.name.split('-')[1])