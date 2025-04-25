# chatbot/consumers.py
import json
import pandas as pd
import os
from channels.generic.websocket import AsyncWebsocketConsumer
from .ai_model import (
    collect_user_skills,
    calculate_skill_gap,
    collect_industry,
    process_user_input,
    further_qna
)

import time 

from .utils import create_skills_model


field_list = ['Data Science', 'E-commerce']
base_dir = os.path.dirname(os.path.abspath(__file__))
jobs_file_path = os.path.join(base_dir, 'Jobs.xlsx')
job_data = pd.read_excel(jobs_file_path, sheet_name='Data Science')  # Đọc dữ liệu từ file Excel
job_list = job_data['Job Title'].tolist()
print(job_list)

class ChatConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        self.session_state = {
            'field': None,
            'job_want': None,
            'customer_data': None,
            'skills_to_ask': [],
            'model': None,
        }
        self.context = ""  # Store only strings
        await self.accept()
        # Send initial greeting
        greeting = "Chào mừng bạn đến với hệ thống! Bạn muốn hoạt động trong lĩnh vực nào?"
        self.context += greeting  # Append the greeting to the context
        await self.send(text_data=json.dumps({"message": greeting}))

    async def disconnect(self, close_code):
        print("Disconnected:", close_code)

    async def receive(self, text_data):
        data = json.loads(text_data)
        message = data.get('message', '').strip()
        response_data = ""

        if message:
            self.context += message  # Append only the user message as a string

            # Notify client that processing has started
            await self.send(text_data=json.dumps({"loading": True}))
            if self.session_state['field'] is None and message in field_list:
                self.session_state['field'] = message
                response_data = f"Lĩnh vực {message} đã được chọn."
            elif self.session_state['field'] and self.session_state['job_want'] is None and message in job_list:
                self.session_state['job_want'] = message
                response_data = f"Vị trí {message} đã được chọn."
            elif self.session_state['field'] and self.session_state['job_want'] and self.session_state['customer_data'] is None:
                self.session_state['customer_data'] = message
                response_data = f"Kỹ năng của bạn đã được ghi nhận: {message}"

            # 2. Hỏi bước tiếp theo ngay sau đó
            if self.session_state['field'] is None:
                response_data += "\nBạn muốn hoạt động trong lĩnh vực nào?"
            elif self.session_state['job_want'] is None:
                response_data += "\nBạn muốn làm vị trí nào?"
            elif self.session_state['customer_data'] is None:
                response_data += "\nBạn có những kỹ năng nào?"

            if (
                self.session_state['field'] is not None and
                self.session_state['job_want'] is not None and
                self.session_state['customer_data'] is not None and
                not self.session_state['skills_to_ask']
            ):
                # Nếu skills_to_ask chưa được thiết lập, gọi collect_industry để lấy danh sách kỹ năng
                skills_to_ask = collect_industry(
                    customer_data=self.session_state['customer_data'],
                    field=self.session_state['field'],
                    job_want=self.session_state['job_want'],
                    customer_id="C00001"
                )
                self.session_state['skills_to_ask'] = skills_to_ask
                time.sleep(5)
                ##### Những kĩ 
                response_data = f"Những kỹ năng cần thiết cho vị trí {self.session_state['job_want']} là: {skills_to_ask}. Bây giờ tôi sẽ hỏi bạn về các kĩ năng cần thiết cho vị trí {self.session_state['job_want']}. Hãy nhấn 'OK' để tiếp tục."    

                # Tạo model Pydantic để lưu kỹ năng
                self.session_state['model'] = create_skills_model(skills_to_ask)
                skills_to_ask.append(skills_to_ask[-1])  # Append the last skill to the list
                self.session_state['skills_to_ask'] = skills_to_ask
            elif self.session_state['skills_to_ask']:
                # Nếu danh sách skills_to_ask đã có, hỏi người dùng về kỹ năng đầu tiên
                current_skill = self.session_state['skills_to_ask'][0]
                skill_response = collect_user_skills(current_skill)
                response_data = skill_response
                self.session_state['skills_to_ask'].remove(current_skill)

                # Nếu đã hỏi hết kỹ năng, chuyển sang xử lý chênh lệch kỹ năng
                if self.session_state['skills_to_ask'] == []:
                    print(skill_response)
                    user_skills = process_user_input(self.context, self.session_state['model'])
                    skill_gap = calculate_skill_gap(user_skills)
                    skill_gap = json.dumps(skill_gap, indent=4, ensure_ascii=False)  # Convert to JSON string
                    response_data = skill_gap

        # Notify client that processing has ended and send the response
        self.context += response_data
        await self.send(text_data=json.dumps({"loading": False, "message": response_data}))