from openai import OpenAI


def get_gpt_response(api_key, message, model="gpt-4-1106-preview"):
    client = OpenAI(api_key=api_key)
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role" : "user",
                "content": message,
            }
        ],
        model=model,
    )
    return chat_completion.choices[0].message.content