import os
import openai
import re
import aiohttp
import json
from google.colab import userdata
OAclient = openai.OpenAI(api_key=userdata.get('openai_key'))

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

async def text_complete_async(prompt):
    async with aiohttp.ClientSession() as session:
        payload = {
            "model": "gpt-3.5-turbo",
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ]
        }
        headers = {
            "Authorization": f"Bearer YOUR_OPENAI_API_KEY",
            "Content-Type": "application/json"
        }

        async with session.post('https://api.openai.com/v1/chat/completions', headers=headers, data=json.dumps(payload)) as response:
            if response.status == 200:
                chat_completion = await response.json()
                return chat_completion['choices'][0]['message']['content']
            else:
                raise Exception(f"OpenAI API request failed with status code: {response.status}")

def descriptor_generate(key_term):
    descriptor_prompt = "Generate a list of 25 descriptors that could be appended onto the phrase: " + key_term + ". The descriptors should be related to the phrase and should be unique from one another. It is required that the list should be a Python list of strings."
    return text_complete(descriptor_prompt)

def normalize_ingredients(ingredients):
    normalize_prompt = "Given the following string representing a list of ingredients, extract the core ingredients from the string. These ingredients should be expressed in one or two word without extra descriptors. Then, return ONLY a string representing a Python list of tuples of the form (core ingredients, quantity number, quantity unit). The tuple should follow the format: ('ingredient', QTY, 'quantity unit'). The quantity number should be a floating-point number. No fractions should be present. The quantity unit is optional and should only be used for ingredients measured in quantities such as grams, milliliters, tablespoons, cups, etc. If no quantity unit is present, return None. : " + ingredients
    return text_complete(normalize_prompt)

def action_extraction(instructions):
    action_extraction_prompt = "Given the following string representing a Python list of recipe instructions, extract the core actions from the string expressed as one or two word items. Then, return a string representing a Python list of the actions. Ensure that the proper brackets are included: " + instructions
    return text_complete(action_extraction_prompt)