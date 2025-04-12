from google import genai
from google.genai import types
from pydantic import BaseModel
import json

# Định nghĩa cấu trúc dữ liệu cho các kỹ năng
class DataScienceModel(BaseModel):
    python: int = 0
    sql: int = 0
    machine_learning: int = 0
    deep_learning: int = 0
    data_visualization: int = 0
    statistics: int = 0
    big_data_tools: int = 0
    etl_data_pipeline: int = 0
    business_knowledge: int = 0
    communication: int = 0
    ab_testing: int = 0
    cloud_platforms: int = 0
    excel: int = 0
    tableau_power_bi: int = 0
    nlp: int = 0
    computer_vision: int = 0
    bioinformatics: int = 0
    time_series_analysis: int = 0
    data_cleaning: int = 0
    storytelling_with_data: int = 0
    problem_solving: int = 0
    critical_thinking: int = 0
    collaboration: int = 0
    project_management: int = 0
    adaptability: int = 0
    # Các trường hợp có thể xảy ra

industry = None
class DSProcessModel(BaseModel):
    python: int
    sql: int
    machine_learning: int
    deep_learning: int
    data_visualization: int
    statistics: int
    big_data_tools: int
    etl_data_pipeline: int
    business_knowledge: int
    communication: int
    ab_testing: int
    cloud_platforms: int
    excel: int
    tableau_power_bi: int
    nlp: int
    computer_vision: int
    bioinformatics: int
    time_series_analysis: int
    data_cleaning: int
    storytelling_with_data: int
    problem_solving: int
    critical_thinking: int
    collaboration: int
    project_management: int
    adaptability: int
    # Các trường hợp có thể xảy ra

# Bộ kỹ năng tham chiếu theo ngành nghề
industry_skill_map = {
    "Data Analyst": DataScienceModel(python=3, sql=4, data_visualization=4, statistics=3),
    "Data Scientist": DataScienceModel(python=4, sql=3, data_visualization=3, statistics=4, machine_learning=4, deep_learning=3),
    "Machine Learning Engineer": DataScienceModel(python=4, sql=3, machine_learning=5, deep_learning=4),
}

# Khởi tạo API key và client Google GenAI
apikey = "AIzaSyCC_s9QxeNr2bOSW_ovqHYdJH65MJIQdow"
client = genai.Client(api_key=apikey)

session_state = {
    'asked_questions': [],  # Các kỹ năng đã hỏi
    'skills_data': DataScienceModel(**{field: 0 for field in DataScienceModel.__annotations__})  # Khởi tạo dữ liệu kỹ năng mặc định
}

def collect_industry(message):
    global industry
    if message in industry_skill_map.keys():
        industry = message
        global required_skills, skills_to_ask
        required_skills = [key for key, value in industry_skill_map.get(industry).model_dump().items() if value != 0]
        # Lấy danh sách kỹ năng cần hỏi
        skills_to_ask = [skill for skill in required_skills]
        return f"Ngành nghề '{industry}' đã được chọn. Vui lòng cung cấp kỹ năng của bạn."
    else:
        return f"Ngành nghề '{message}' không được hỗ trợ."


def collect_user_skills(skill_to_ask):
    global industry
    print(f"Đang hỏi về kỹ năng: {skill_to_ask}")
    system_instruction = f"""
            Bạn là một hệ thống AI thu thập thông tin về kỹ năng. Hãy hỏi người dùng về mức độ thành thạo của họ với kỹ năng '{skill_to_ask}'.
            Câu hỏi ngắn gọn và phải chứa thông tin về {skill_to_ask}.
            Hãy sử dụng thang điểm từ 1 đến 5 để đánh giá mức độ thành thạo của họ.
            1. Không biết
            2. Biết một chút
            3. Biết
            4. Biết nhiều
            5. Thành thạo
            Hãy tôn trọng quyền riêng tư và bảo mật của người dùng.
            Hãy hỏi người dùng về mức độ thành thạo của họ với từng kỹ năng một.
            Hãy đảm bảo rằng bạn không hỏi quá nhiều câu hỏi một lúc và hãy lắng nghe phản hồi của người dùng.
        """
    response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents='',  # Use the 'message' parameter here
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                # response_mime_type='application/json',  
            )
        )
    return response.text

    
def process_user_input(message):
    system_instruction = f"""
        Bạn là một hệ thống AI phân tích về kĩ năng. 
        Bạn sẽ thu thập thông tin người dùng về các kĩ năng của họ.
        Bạn hãy chuyển ngôn ngữ của người dùng, đánh giá kĩ năng của họ và đưa về thang mức độ từ 1 đến 5
            1. Không biết
            2. Biết một chút
            3. Biết
            4. Biết nhiều
            5. Thành thạo
        Dưới đây là mô tả người dùng:
        {message}
    """
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=message,
        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            response_mime_type='application/json',
            response_schema=DSProcessModel,
        )
    )
    return response.text 

# Hàm xử lý chênh lệch kỹ năng
def calculate_skill_gap(user_skills, required_skills):
    skill_gap = {}
    user_skills = json.loads(user_skills)
    for skill, required_level in required_skills.items():
        if required_level != 0:
            user_level = user_skills.get(skill, 0)
            skill_gap[skill] = required_level - user_level
    return skill_gap

# Hàm hiển thị chênh lệch kỹ năng
def display_skill_gap(skill_gap):
    result = "Chênh lệch kỹ năng:\n"
    for skill, gap in skill_gap.items():
        result += f"- {skill}: {'Đạt yêu cầu' if gap <= 0 else f'Cần cải thiện {gap} điểm'}\n"
    return result

