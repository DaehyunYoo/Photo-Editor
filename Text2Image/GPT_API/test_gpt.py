from openai import OpenAI
import os

api_key = os.environ['OPENAI_API_KEY']

client = OpenAI(api_key=api_key)
model = "gpt-3.5-turbo"

query = 'ChatGPT는 어디 에 활용될 수 있나요?'
message = [{'role': 'user', 'content': query}]
completion = client.chat.completions.create(model=model, messages=message)
response_text = completion.choices[0].message.content
 
print(response_text)