from google import genai
from django.http import JsonResponse

client = genai.Client(api_key="AIzaSyCC_s9QxeNr2bOSW_ovqHYdJH65MJIQdow")


content = [
    {
        "role": "system",
        "content": "You are a helpful assistant that speaks like a human.",
    },
    {
        "role": "user",
        "content": "Help me with a skill comparison.",
    },
    {
        "role": "assistant",
        "content": "Help user with skill comparison.",
    },
]

def get_ai_response(message):
    response = client.models.generate_content(
        model="gemini-2.0-flash", contents=message,
    )
    return response.text
