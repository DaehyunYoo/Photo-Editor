import openai
import os

api_key = os.environ['OPENAI_API_KEY']

client = openai.OpenAI(api_key=api_key)
assistant = client.beta.assistants.retrieve(assistant_id='asst_DPIWgRQEsfg4hilmf1pj8bim')
# print(assistant)

thread = client.beta.threads.create()

message = client.beta.threads.messages.create(
  thread_id=thread.id,
  role="user",
  content="고양이 그려줘"
)

run = client.beta.threads.runs.create_and_poll(
  thread_id=thread.id,
  assistant_id=assistant.id,
#   instructions="개 그려줘"
)

if run.status == 'completed': 
  messages = client.beta.threads.messages.list(
    thread_id=thread.id
  )
  print(messages.data[0].content[0].text.value)
else:
  print(run.status)