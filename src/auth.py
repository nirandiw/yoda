from openai import OpenAI
import os
from dotenv import load_dotenv

def get_openai_client():
    load_dotenv()
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    return client
    