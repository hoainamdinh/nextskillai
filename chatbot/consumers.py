# chatbot/consumers.py
import json
from channels.generic.websocket import AsyncWebsocketConsumer
from .ai_model import get_ai_response  # model AI xử lý câu trả lời

class ChatConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        await self.accept()

    async def disconnect(self, close_code):
        print("Disconnected:", close_code)

    async def receive(self, text_data):
        data = json.loads(text_data)
        message = data.get('message', '')
        
        # Xử lý AI
        response = get_ai_response(message)

        await self.send(text_data=json.dumps({
            'response': response
        }))
