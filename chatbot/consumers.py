# chatbot/consumers.py
import json
from channels.generic.websocket import AsyncWebsocketConsumer
from .ai_model import (
    collect_user_skills,
    calculate_skill_gap,
    display_skill_gap,
    DataScienceModel,
    industry_skill_map,
    collect_industry,
    process_user_input
)

class ChatConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        self.session_state = {
            'industry': None,
            'skills_data': DataScienceModel(),
            'skills_to_ask': [],
        }
        await self.accept()

    async def disconnect(self, close_code):
        print("Disconnected:", close_code)

    context = ''

    async def receive(self, text_data):
        data = json.loads(text_data)
        message = data.get('message', '').strip()
        response_data = {}
        self.context += (message+", ") if message else None
        # If the industry is not selected
        if self.session_state['industry'] is None:
            industry_response = collect_industry(message)

            if "đã được chọn" in industry_response:
                # Valid industry selected
                self.session_state['industry'] = message
                self.session_state['skills_to_ask'] = [key for key, value in industry_skill_map.get(message).model_dump().items() if value != 0]
                response_data['industry_response'] = industry_response
            else:
                # Invalid industry
                response_data['industry_response'] = industry_response

        
            # Industry is already selected, start asking skills
        if self.session_state['skills_to_ask']:
            print(self.session_state['skills_to_ask'])
            current_skill = self.session_state['skills_to_ask'][0]
            skill_response = collect_user_skills(current_skill)
            response_data['skills_response'] = skill_response
            self.session_state['skills_to_ask'].pop(0)
                # Check if the user provided a valid skill level
        else:
            print(self.context)
            user_skills = process_user_input(self.context)
            # All skills have been asked
            required_skills = {key: value for key, value in industry_skill_map.get(self.session_state['industry']).model_dump().items() if value != 0}
            print(user_skills)
            skill_gap = calculate_skill_gap(user_skills, required_skills)
            result = display_skill_gap(skill_gap)
            response_data['final_result'] = result

        # Send the response back to the client
        await self.send(text_data=json.dumps(response_data))
