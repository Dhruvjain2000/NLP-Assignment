import os
from openai import OpenAI

class ChatGPT:
    def __init__(self, api_key):
        self.api_key = api_key

    def ask(self, prompt):
        client = OpenAI(api_key=self.api_key)

        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model="gpt-3.5-turbo",
        )

        data = chat_completion.choices[0]['text']
        return data

# Usage example:
# chatgpt = ChatGPT(api_key='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')
# response = chatgpt.ask("What is the meaning of life?")
# print(response)
