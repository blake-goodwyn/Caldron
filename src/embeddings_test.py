import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import openai
from whitespace_correction import WhitespaceCorrector
import ast
import nlp

# Set your OpenAI API key
openai.api_key = 'your-api-key'  # Replace with your actual API key

target = 'recipes-2024-02-26-1534.csv'
quantity_units = {'tsp', 'teaspoon', 'teaspoons', 'tbsp', 'tablespoon', 'tablespoons', 'cup', 'cups', 'c', 'ml', 'milliliter', 'milliliters', 'liter', 'liters', 'l', 'gram', 'grams', 'g', 'kilogram', 'kilograms', 'kg', 'oz', 'ounce', 'lb', 'pound'}
non_ingredient_keywords = {'ripe', 'softened', 'room', 'temperature', 'ground', 'fresh', 'dried', 'chopped', 'sliced', 'of', 'at', 'with', 'for', 'and'}

def preprocess_list(ingredients):
    ing_list = ast.literal_eval(ingredients)
    assert isinstance(ing_list, list)
    ing_list = [cor.correct_text(ing).lower() for ing in ing_list]
    return ' , '.join(ing_list)

def extract_ingredient(phrase):
    #print("RAW: ", phrase)
    phrase = cor.correct_text(phrase).lower()
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

def process_ingredient_list(ingredients):
    processed_ingredients = []
    ing_list = ast.literal_eval(ingredients)
    assert isinstance(ing_list, list)
    for ingredient in ing_list:
        main_ingredient = extract_ingredient(ingredient)
        processed_ingredients.append(main_ingredient)

    return processed_ingredients

# Function to generate embeddings
def generate_embeddings(text):
    response = openai.Embedding.create(
        input=[text],
        engine="text-similarity-babbage-001",  # You can choose different engines as needed
    )
    # The response includes embeddings; extract them
    embedding = response['data'][0]['embedding']
    return embedding

# Visualization
def plot_with_pca(embeddings):
    pca = PCA(n_components=2)
    result = pca.fit_transform(embeddings)
    plt.scatter(result[:, 0], result[:, 1])
    plt.show()

def plot_with_tsne(embeddings):
    tsne = TSNE(n_components=2)
    result = tsne.fit_transform(embeddings)
    plt.scatter(result[:, 0], result[:, 1])
    plt.show()

# Load your DataFrame
df = pd.read_csv(target)  # Replace with your dataset file

# Preprocess your DataFrame
cor = WhitespaceCorrector.from_pretrained()
df['ingredients'] = df['ingredients'].apply(preprocess_list)

# Add a column for embeddings
df['ingredients_EMB'] = df['ingredients'].apply(generate_embeddings)
df['instructions_EMB'] = df['instructions'].apply(generate_embeddings)
df['process_ingredients'] = df['ingredients'].apply(process_ingredient_list)

# Analysis Functions
def calculate_ingredient_frequency(df, Counter):
    for i in df['process_ingredients']:
        for j in i:
            Counter[j] += 1

ing_Counter = Counter()

# Apply Analysis Functions
ingredient_freq = calculate_ingredient_frequency(df, ing_Counter)

# Convert embeddings list to numpy array for visualization
embeddings_array = np.array(df['embeddings'].tolist())

# Visualize with PCA
plot_with_pca(embeddings_array)

# Visualize with t-SNE
plot_with_tsne(embeddings_array)