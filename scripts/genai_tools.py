import os
import openai
import asyncio
import backoff
import requests
import ast
import tiktoken
from asyncio import Semaphore
from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")
OAclient = openai.OpenAI()
OAClientAsync = openai.AsyncOpenAI()
tokenizer = tiktoken.encoding_for_model('gpt-3.5-turbo')

MAX_CALLS_PER_MINUTE = 500
SEM = Semaphore(MAX_CALLS_PER_MINUTE)
TC_MODEL = "gpt-3.5-turbo"
EMB_MODEL = "text-embedding-3-small"

#####

async def limited_call(func, *args, **kwargs):
    """ Make a limited call to a function using a semaphore. """
    async with SEM:
        await asyncio.sleep(60/MAX_CALLS_PER_MINUTE)
        result = await asyncio.to_thread(func, *args, **kwargs)
        return result

@backoff.on_exception(backoff.constant, (requests.exceptions.RequestException, openai.RateLimitError, openai.OpenAIError), max_time=3600)
def text_complete(prompt, client=OAclient, max_tokens=16000):
    try:
        
        encoded_prompt = tokenizer.encode(prompt)
        if len(encoded_prompt) > max_tokens:
            encoded_prompt = encoded_prompt[:max_tokens]  # Keep the last max_tokens tokens

        truncated_prompt = tokenizer.decode(encoded_prompt)
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": truncated_prompt,
                }
            ],
            model=TC_MODEL,
        )
        return chat_completion.choices[0].message.content
    except openai.RateLimitError as e:
        # Handle the specific rate limit error
        #print("Rate limit exceeded, retrying...")
        raise
    except openai.OpenAIError as e:
        # Handle other OpenAI errors
        print(f"OpenAI error: {e}")
        raise
    except requests.exceptions.RequestException as e:
        # Handle requests-related errors
        print(f"Request error: {e}")
        raise

def get_embedding(text):
    return OAclient.embeddings.create(input = [text], model=EMB_MODEL).data[0].embedding

@backoff.on_exception(backoff.expo, (requests.exceptions.RequestException,openai.RateLimitError, openai.OpenAIError), max_time=300)
async def text_complete_async(prompt) -> None:
    try:
        chat_completion = await OAClientAsync.chat.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model=TC_MODEL,
        )
        return chat_completion.choices[0].message.content
    except openai.RateLimitError as e:
        # Handle the specific rate limit error
        print("Rate limit exceeded, retrying...")
        raise
    except openai.OpenAIError as e:
        # Handle other OpenAI errors
        print(f"OpenAI error: {e}")
        raise
    except requests.exceptions.RequestException as e:
        # Handle requests-related errors
        print(f"Request error: {e}")
        raise

async def get_embedding_async(text) -> None:
    text = text.replace("\n", " ")
    return (await OAClientAsync.embeddings.create(input = [text], model=EMB_MODEL)).data[0].embedding

def descriptor_generate(key_term):
    descriptor_prompt = "Generate a list of 50 descriptors that could be appended onto the phrase: " + key_term + ". The descriptors should be related to the phrase, should be varied, and should be unique from one another. It is required that the list should be a Python list of strings."
    temp = text_complete(descriptor_prompt)

    while True:
        try:
            l = ast.literal_eval(temp)
            if not isinstance(l, list):
                raise ValueError("Result is not a list")

            for i in l:
                if not isinstance(i, str):
                    raise ValueError("List item is not a string")
            break  # Break the loop if no exceptions are raised

        except (ValueError, SyntaxError) as e:
            print(f"Error: {e}")
            # Re-prompt and continue the loop asynchronously
            temp = text_complete(descriptor_prompt)

    return l

def standardize_ingredients(ingredients):
    normalize_prompt = "Given the following string representing a list of ingredients (and potentially utensils or supplies), extract the core ingredients from the string. These ingredients should be expressed in one or two word phrases without extra descriptors. Then, return ONLY a string representing a Python list of tuples of the form (core ingredients, quantity number, quantity unit). The tuple should follow the format: ('ingredient', QTY, 'quantity unit'). The quantity number should be a floating-point number. No fractions should be present. The quantity unit is optional and should only be used for ingredients measured in quantities such as grams, milliliters, tablespoons, cups, etc. If no quantity unit is present, return None. : \n\n" + ingredients
    return text_complete(normalize_prompt)

def standardize_instructions(ingredients):
    normalize_prompt = "Given the following string representing a list of instructions (and potentially utensils or supplies), extract the core instructions from the string. Then, return ONLY a string representing a Python list of strings with each string representing a discrete action. : \n\n" + ingredients
    return text_complete(normalize_prompt)

action_extraction_prompt = "Given the following string representing a some number of recipe instructions, extract the actions in sequence from the string. Each action should be a single, distinct verb. For example, 'Add the eggs and whisk for 15 seconds' should be something like ['add', 'whisk']. Return a Python list of these actions in sequence. Ensure that the proper brackets are included: \n\n"

def actions_extraction(instructions):
    #print(f"Extracting actions from the following instructions: {instructions}")
    prompt = action_extraction_prompt + instructions
    temp = text_complete(prompt)

    while True:
        try:
            l = ast.literal_eval(temp)
            if not isinstance(l, list):
                raise ValueError("Result is not a list")

            for i in l:
                if not isinstance(i, str):
                    raise ValueError("List item is not a string")
            break  # Break the loop if no exceptions are raised

        except (ValueError, SyntaxError) as e:
            print(f"Error: {e}")
            # Re-prompt and continue the loop asynchronously
            temp = text_complete(prompt)

    return l

def extract_action(action):
    prompt = "Given the following string representing a step in a recipe, extract the core action from the string. Each action should be a single, distinct verb. For example, 'Add the eggs and whisk for 15 seconds' should be something like 'Whisk'. Return as a single word: \n\n" + action
    return text_complete(prompt)

def cluster_label(prompt):
    prompt = "Given the following string representing a list of actions associated with specific phase of a recipe, examine the actions and determine a one token label for the category of the list. Consider labels that repeat prominently in this determination. The string is formatted in lines of the position of the action in the recipe's sequence of action and then a label for the action. Return a single word to be used as a label for the cluster \n\n" + prompt
    if len(tokenizer.encode(prompt)) > 16385: #openapi token limit
        legal_tokens = tokenizer.encode(prompt)[:-10]
        prompt = tokenizer.decode(legal_tokens)
    
    return text_complete(prompt)

async def action_extraction_async(recipe):
    assert(type(recipe) == dict)
    prompt = action_extraction_prompt + recipe['instructions']
    print(str(recipe['index']+1), " | PROMPT!")
    chat_completion = await OAClientAsync.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model="gpt-3.5-turbo",
    )
    recipe["instr_list"] = chat_completion.choices[0].message.content