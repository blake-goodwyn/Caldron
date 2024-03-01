import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from itertools import combinations

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
    for index, row in df.iterrows():
        try:
            ingredients = eval(row['Processed Ingredients'])
            flour_cups = 0
            normalized_ingredients = []

            # First, find the quantity of flour in cups
            for ingredient in ingredients:
                name, quantity, unit = ingredient
                if "all purpose flour" or "flour" in name.lower():
                    flour_cups = normalize_to_flour(quantity, unit)
                    break

            # Skip normalization if flour is not found or has zero quantity
            if flour_cups == 0:
                continue

            # Normalize other ingredients against the quantity of flour
            for ingredient in ingredients:
                name, quantity, unit = ingredient
                if name != "all purpose flour":
                    quantity_in_cups = normalize_to_flour(quantity, unit)
                    normalized_quantity = quantity_in_cups / flour_cups
                    normalized_ingredients.append((name, normalized_quantity, 'cups'))

            # Store the normalized ingredients
            df.at[index, 'Normalized Ingredients'] = str(normalized_ingredients)
        except Exception as e:
            pass
            #print(e)
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

def visualize(file):
    # Load the data
    df = normalize_ingredients(pd.read_csv(file))
    co_occurrence = Counter()

    # Process the ingredient data
    ingredient_data = {}
    for index, row in df.iterrows():
        try:
            ingredients = eval(row['Normalized Ingredients'])  # Replace with actual column name
            co_occurrence = update_co_occurrence(ingredients, co_occurrence)
            for ingredient in ingredients:
                name, quantity, unit = ingredient
                if name not in ingredient_data:
                    ingredient_data[name] = {'total_quantity': 0, 'recipe_count': 0}
                ingredient_data[name]['total_quantity'] += quantity  # Adjust if unit conversion is needed
                ingredient_data[name]['recipe_count'] += 1
        except Exception as e:
            pass
            #print(e)

    # Prepare data for plotting
    names = []
    quantities = []
    prevalences = []
    for name, data in ingredient_data.items():
        names.append(name)
        quantities.append(data['total_quantity']/data['recipe_count'])
        prevalences.append(data['recipe_count'])

    #Filter for top ingredients
    k=20
    names = names[:k]
    quantities = quantities[:k]
    prevalences = prevalences[:k]

    # Create the graph
    plt.figure(figsize=(10, 6))
    plt.scatter(quantities, prevalences, alpha=0.5)  # Adjust size scaling as needed
    plt.xlabel('Relative Quantity (Cups)')
    plt.ylabel('Recipe Prevalence')
    plt.xticks(rotation=45)
    plt.title('Ingredient Prevalence and Quantity in Tart Recipes')

    ing_pos={}

    # Add ingredient names as labels
    for i, txt in enumerate(names):
        ing_pos[txt] = (quantities[i], prevalences[i])
        plt.annotate(txt, (quantities[i], prevalences[i]), rotation=45)

    # Draw lines based on co-occurrence
    min_co_occurrence = 20  # Set your threshold

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

    plt.show()

visualize('C:/Users/blake/Documents/GitHub/ebakery/data/tart/processed-tart-recipe-2024-02-29-2229.csv')