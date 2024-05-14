import os
import pandas as pd
from sqlalchemy import create_engine, Column, Integer, String, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Define the SQLAlchemy ORM base class
Base = declarative_base()

# Define the Recipe class mapping to the recipes table
class Recipe(Base):
    __tablename__ = 'recipes'
    
    id = Column(Integer, primary_key=True)
    url = Column(String)
    name = Column(String)
    instructions = Column(Text)
    processed_ingredients = Column(Text)

# Create an SQLite engine and bind it to the metadata of the base class
engine = create_engine('sqlite:///recipes.db')
Base.metadata.create_all(engine)

# Establish a session
Session = sessionmaker(bind=engine)
session = Session()

# Function to read CSV files and add data to the database
def add_recipes_from_csv(directory_path):
    for filename in os.listdir(directory_path):
        if filename.endswith(".csv"):
            file_path = os.path.join(directory_path, filename)
            # Read the CSV file into a DataFrame
            df = pd.read_csv(file_path)
            
            # Iterate over DataFrame rows
            for index, row in df.iterrows():
                # Create a new Recipe object
                recipe = Recipe(
                    id=row['ID'],
                    url=row['URL'],
                    name=row['Recipe Name'],
                    instructions=row['Instructions'],
                    processed_ingredients=row['Processed Ingredients']
                )
                # Add the Recipe object to the session
                try:
                    session.add(recipe)
                except Exception as e:
                    print(f"Error adding recipe from {filename}: {e}")
    
    # Commit the session to save all objects
    session.commit()

# Example usage
directory_path = 'C:/Users/blake/Documents/GitHub/ebakery/data/GOOD DATASETS'
add_recipes_from_csv(directory_path)

# Close the session
session.close()
