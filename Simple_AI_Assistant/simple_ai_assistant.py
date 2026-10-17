from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

client = OpenAI(
    base_url=os.get_env("BASE_URL"),
    api_key=os.get_env("API_KEY")
)

response = client.chat.completions.create(
    model=os.get_env("MODEL"),
    messages=[
        {
            "role": "user",
            "content": "Hello LLM!"
        }
    ]
)

print("LLM Response:")
print(response.choices[0].message.content)
