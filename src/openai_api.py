from dotenv import load_dotenv
from openai import OpenAI
import os


def send_request(input, model="gpt-4.1-mini", temperature=0.5, max_tokens=1000, api_key=None):
    
    # Send the request to the API
    client = OpenAI(api_key=api_key)
    try:
        response = client.responses.create(
            model=model,
            input=input,
            temperature=temperature,
            max_output_tokens=max_tokens
            )
        if response.status == "completed":
            return 200, response
        else:
            return 500, None
    except Exception:
        return 500, None


def build_prompt(prompt, instructions):

    input = [
        {
            "role": "system",
            "content": instructions
        },
        {
            "role": "user",
            "content": prompt
        }
    ]

    return input