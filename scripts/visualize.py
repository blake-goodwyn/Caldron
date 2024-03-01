import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

# Define a conversion function to standardize measurements to cups
def normalize_to_flour(quantity, unit):
    conversion_factors = {
        'g': 0.00422675,  # Grams to cups (for flour, approximately)
        'tablespoons': 0.0625,  # Tablespoons to cups
        'teaspoon': 0.0208333,  # Teaspoons to cups
        # ... other units
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
            print(e)
    return df

def visualize(file):
    # Load the data
    df = normalize_ingredients(pd.read_csv(file))

    # Process the ingredient data
    ingredient_data = {}
    for index, row in df.iterrows():
        try:
            ingredients = eval(row['Normalized Ingredients'])  # Replace with actual column name
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

    # Create the bubble chart
    plt.figure(figsize=(10, 6))
    plt.scatter(quantities, prevalences, alpha=0.5)  # Adjust size scaling as needed
    plt.xlabel('Total Quantity')
    plt.ylabel('Recipe Prevalence')
    plt.xticks(rotation=45)
    plt.title('Ingredient Prevalence and Quantity in Tart Recipes')

    # Add ingredient names as labels
    for i, txt in enumerate(names):
        plt.annotate(txt, (quantities[i], prevalences[i]), rotation=45)

    plt.show()

visualize('C:/Users/blake/Documents/GitHub/ebakery/data/banana-bread/processed-banana-bread-recipes-2024-02-27-1547.csv')