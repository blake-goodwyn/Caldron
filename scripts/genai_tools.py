import os
import openai
import backoff
import re
import asyncio
import aiohttp
import json
from dotenv import load_dotenv
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

sem  = asyncio.Semaphore(100)

OAclient = openai.OpenAI()
OAClientAsync = openai.AsyncOpenAI()

def text_complete(prompt, client=OAclient):
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model="gpt-3.5-turbo",
    )
    return chat_completion.choices[0].message.content

def get_embedding(text, model="text-embedding-3-small"):
    return OAclient.embeddings.create(input = [text], model=model).data[0].embedding


async def text_complete_async(prompt) -> None:
    chat_completion = await OAClientAsync.chat.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model="gpt-3.5-turbo",
    )
    return chat_completion.choices[0].message.content

@backoff.on_exception(backoff.expo, openai.RateLimitError)
async def get_embedding_async(text, model="text-embedding-3-small") -> None:
    text = text.replace("\n", " ")
    return (await OAClientAsync.embeddings.create(input = [text], model=model)).data[0].embedding

def descriptor_generate(key_term):
    descriptor_prompt = "Generate a list of 50 descriptors that could be appended onto the phrase: " + key_term + ". The descriptors should be related to the phrase, should be varied, and should be unique from one another. It is required that the list should be a Python list of strings."
    return text_complete(descriptor_prompt)

def standardize_ingredients(ingredients):
    normalize_prompt = "Given the following string representing a list of ingredients (and potentially utensils or supplies), extract the core ingredients from the string. These ingredients should be expressed in one or two word phrases without extra descriptors. Then, return ONLY a string representing a Python list of tuples of the form (core ingredients, quantity number, quantity unit). The tuple should follow the format: ('ingredient', QTY, 'quantity unit'). The quantity number should be a floating-point number. No fractions should be present. The quantity unit is optional and should only be used for ingredients measured in quantities such as grams, milliliters, tablespoons, cups, etc. If no quantity unit is present, return None. : \n\n" + ingredients
    return text_complete(normalize_prompt)

def standardize_instructions(ingredients):
    normalize_prompt = "Given the following string representing a list of instructions (and potentially utensils or supplies), extract the core instructions from the string. Then, return ONLY a string representing a Python list of strings with each string representing a discrete action. : \n\n" + ingredients
    return text_complete(normalize_prompt)

def action_extraction(instructions):
    action_extraction_prompt = "Given the following string representing a Python list of recipe instructions, extract the core actions from the string expressed as one or two word items. Then, return a string representing a Python list of the actions. Ensure that the proper brackets are included: \n\n" + instructions
    return text_complete(action_extraction_prompt)