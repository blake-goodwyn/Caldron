import requests
import os
from dotenv import load_dotenv
import pandas as pd
from sqlalchemy import create_engine, Column, Integer, String, JSON, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from ratelimit import limits, sleep_and_retry
from tqdm import tqdm

load_dotenv()

_DEBUG = False
CALLs = 1000  # Limited to 1000 API calls per hour
RATE_LIMIT = 3600  # 3600 seconds = 1 hour

cwd = os.getcwd()
output_directory = os.path.join(cwd, 'ingredients')
fdc_api_key = os.getenv("FoodData_Central_API_KEY")

# SQLAlchemy setup
Base = declarative_base()

class Ingredient(Base):
    __tablename__ = 'ingredients'
    fdcId = Column(Integer, primary_key=True)
    description = Column(String)
    foodCategory = Column(String)
    foodNutrients = Column(JSON)
    finalFoodInputFoods = Column(JSON)
    foodMeasures = Column(JSON)
    servingSizeUnit = Column(String)
    servingSize = Column(Float)

engine = create_engine('sqlite:///' + os.path.join(output_directory, 'ingredients_db.sqlite'))
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)
session = Session()

@sleep_and_retry
@limits(calls=CALLs, period=RATE_LIMIT)
def search_food(api_key, query):
    url = "https://api.nal.usda.gov/fdc/v1/foods/search"
    params = {
        "api_key": api_key,
        "query": query
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        foods = response.json().get("foods", [])
        df = pd.DataFrame(foods)
        
        #Pare down the dataframe to only the columns we want
        try:
            df = df[['fdcId','description','foodCategory','servingSizeUnit','servingSize','foodNutrients','finalFoodInputFoods', 'foodMeasures']]
        except Exception as e:
            try:
                df = df[['fdcId','description','foodCategory','foodNutrients', 'finalFoodInputFoods', 'foodMeasures']]
            except:    
                print(f"Error: {e}")
                return None

        return df
    else:
        return None

def update_ingredients_db(fdc_api_key, term):
    df = search_food(fdc_api_key, term)
    if df is not None:
        for _, row in df.iterrows():
            if not session.query(Ingredient).filter(Ingredient.fdcId == row['fdcId']).first():
                try:
                    ingredient = Ingredient(**row)
                    session.add(ingredient)
                except Exception as e:
                    print(f"Error: {e}")
        session.commit()

def get_nutrient_profile(fdcID, opt="LOCAL"):
    if opt == "LOCAL":
        try:
            ingredient = session.query(Ingredient).filter(Ingredient.fdcId == fdcID).first()
            if ingredient:
                return ingredient.foodNutrients
        except Exception as e:
            print(f"Error: {e}")
    else:
        url = f"https://api.nal.usda.gov/fdc/v1/food/{fdcID}"
        params = {"api_key": fdc_api_key}
        response = requests.get(url, params=params)
        if response.status_code == 200:
            return response.json()
    return None

def update_from_node_graph(file):
    with open(file, 'r') as f:
        search_terms = f.readlines()

    for term in tqdm(search_terms):
        term = term.strip()
        update_ingredients_db(fdc_api_key, term)