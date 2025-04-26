from google import genai
from google.genai import types
from pydantic import BaseModel
import json
import pandas as pd
import os
from .utils import Recommendation_model

# Định nghĩa cấu trúc dữ liệu cho các kỹ năng


# # Khởi tạo API key và client Google GenAI
# apikey = "AIzaSyCC_s9QxeNr2bOSW_ovqHYdJH65MJIQdow"
apikey = 'AIzaSyDtmMJ8WeUZsgYpkpN3vzy70AoSmynzGOQ'
client = genai.Client(api_key=apikey)


def collect_industry(customer_data, field, job_want, customer_id="C00001"):
    required_skills = Recommendation_model(
        CustomerData=customer_data,
        Field=field,
        Jobwant=job_want,
        CustomerID=customer_id,
        model="Miss_skill")
        # Lấy danh sách kỹ năng cần hỏi
    return required_skills


def collect_user_skills(skill_to_ask):
    print(f"Đang hỏi về kỹ năng: {skill_to_ask}")
    system_instruction = f"""
        Bạn là một hệ thống AI thu thập thông tin về kỹ năng. 
        Nhiệm vụ của bạn là hỏi người dùng về mức độ thành thạo kỹ năng '{skill_to_ask}' bằng một câu ngắn gọn, rõ ràng.

        Khi hỏi, hãy mô tả chi tiết các mức độ thành thạo sau:
        1. Không biết: Không có kiến thức hoặc kinh nghiệm.
        2. Biết một chút: Hiểu biết cơ bản hoặc từng tiếp xúc.
        3. Biết: Có thể sử dụng ở mức trung bình.
        4. Biết nhiều: Thành thạo trong các tình huống thông thường.
        5. Thành thạo: Sử dụng xuất sắc và chuyên nghiệp.

        Đồng thời, hãy giải thích ngắn gọn về kỹ năng '{skill_to_ask}' kèm ví dụ thực tế về kĩ năng, công cụ sử dụng từ cơ bản đến nâng cao.

        Ghi nhớ:
        - Tôn trọng quyền riêng tư và bảo mật thông tin cá nhân.
        - Chỉ hỏi từng kỹ năng một lần.
        - Không hỏi quá nhiều nội dung cùng lúc.
        - Chờ phản hồi từ người dùng trước khi hỏi tiếp.
    """
    response = client.models.generate_content(
            model="gemini-2.0-flash-lite-001",
            contents='',  # Use the 'message' parameter here
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
            )
        )
    return response.text
    

def process_user_input(message, model):
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
            response_schema=model,
        )
    )
    return response.text 

# Hàm xử lý chênh lệch kỹ năng
def calculate_skill_gap(user_skills):
    skill_gap = {}
    user_skills = json.loads(user_skills)
    
    # Generate system instruction for AI
    system_instruction = """
        Bạn là một hệ thống AI phân tích chênh lệch kỹ năng. 
        Dựa trên kỹ năng người dùng cung cấp, hãy đánh giá mức độ chênh lệch và đưa ra gợi ý cải thiện.
        Đối với mỗi kỹ năng, nếu mức độ chưa đạt yêu cầu, hãy đề xuất cách cải thiện cụ thể.
    """
    # Prepare input for AI
    input_data = {
        "user_skills": user_skills
    }
    
    # Use AI client to generate recommendations
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=json.dumps(input_data),
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            response_mime_type='application/json'
        )
    )
    return response.text    

    # # Parse AI response
    # recommendations = json.loads(response.text)
    
    # # Process recommendations and calculate skill gap
    # for skill, details in recommendations.items():
    #     required_level = details.get("required_level", 0)
    #     user_level = user_skills.get(skill, 0)
    #     if required_level > user_level:
    #         skill_gap[skill] = {
    #             "gap": required_level - user_level,
    #             "recommendation": details.get("recommendation", "No recommendation available")
    #         }
    
    # return skill_gap

# Hàm hiển thị chênh lệch kỹ năng
def further_qna(message, context):
    """
    Function to generate a Q&A session for a specific skill based on the provided context.
    
    Args:
        skill_name (str): The name of the skill to inquire about.
        context (str): The context or additional information to guide the Q&A session.
    
    Returns:
        str: The AI-generated response for the Q&A session.
    """
    system_instruction = f"""
    Bạn là một hệ thống AI hỗ trợ người dùng cải thiện kỹ năng cá nhân. 
    Người dùng đang muốn cải thiện kỹ năng: {message}. 

    Yêu cầu:
    1. Phân tích kỹ năng thành các kỹ năng con (sub-skills) cần nắm vững.
    2. Gợi ý lộ trình học chi tiết cho từng kỹ năng con (theo cấp độ từ cơ bản → nâng cao).
    3. Đưa ra các nguồn học cụ thể: khóa học online, website, sách hoặc nền tảng miễn phí.
    4. Gợi ý từ khóa tìm kiếm bằng tiếng Anh và tiếng Việt để người dùng dễ tự học trên Google/YouTube.
    5. Giữ câu trả lời ngắn gọn, trực quan và dễ hành động. Không viết chung chung, không dùng câu rườm rà.
    """
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents='',
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
        )
    )
    return response.text



# ################################

import re  # Import required libraries
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings
warnings.filterwarnings("ignore")
import os 
from typing import List
base_dir = os.path.dirname(os.path.abspath(__file__))
jobs_path = os.path.join(base_dir, 'Jobs.xlsx')
skill_path = os.path.join(base_dir, 'Skill.xlsx')

def Pre(df):
    # Lấy tên cột cuối cùng
    last_column = df.columns[-1]
    
    # Hàm xử lý văn bản
    def clean_text(text):
        # Chuyển thành chữ thường
        text = text.lower()
        # Thay thế các ký tự đặc biệt bằng khoảng trắng
        text = text.replace("'", " ")
        text = text.replace('"', " ")
        text = text.replace("(", " ")
        text = text.replace(")", " ")
        text = text.replace("[", " ")
        text = text.replace("]", " ")
        text = text.replace("{", " ")
        text = text.replace("}", " ")
        text = text.replace("!", " ")
        text = text.replace("?", " ")
        text = text.replace(":", " ")
        text = text.replace(";", " ")
        text = text.replace(",", " ")
        text = text.replace(".", " ")
        text = text.replace("-", " ")
        text = text.replace("_", " ")
        text = text.replace("+", " ")
        text = text.replace("=", " ")
        text = text.replace("*", " ")
        text = text.replace("&", " ")
        text = text.replace("^", " ")
        text = text.replace("%", " ")
        text = text.replace("$", " ")
        text = text.replace("#", " ")
        text = text.replace("@", " ")
        text = text.replace("~", " ")
        text = text.replace("`", " ")
        text = text.replace("|", " ")
        text = text.replace("<", " ")
        text = text.replace(">", " ")
        text = text.replace("/", " ")
        text = text.replace("\\", " ")
        return text
    
    # Áp dụng xử lý văn bản cho cột cuối cùng
    df[last_column] = df[last_column].apply(clean_text)
    return df

def Recommendation_model(CustomerData, Field, Jobwant, CustomerID="C00001", model="Miss_skill"):
    # Load data from Excel files
    df_job = pd.read_excel(jobs_path, sheet_name=Field)  # Update with the correct path
    df_skill = pd.read_excel(skill_path)  # Update with the correct path

    # Remove duplicate skills
    df_skill = df_skill.drop_duplicates(subset='Skill', keep='first').reset_index(drop=True)

    # Create skill description column
    df_skill['Skill_Description'] = df_skill['Skill'] + " " + df_skill['Description'] + df_skill['Skill'] + df_skill['Skill']

    # Preprocess data using custom Pre function
    df_job = Pre(df_job)
    df_skill = Pre(df_skill)

    # Function to vectorize skills using TF-IDF
    def vectorize_tfidf(df_skills):
        tfidf_vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf_vectorizer.fit_transform(df_skills['Skill_Description'])
        return tfidf_matrix, tfidf_vectorizer

    # Function to recommend skills based on customer data
    def recommend(customer_data, tfidf_matrix, tfidf_vectorizer, df_skill, threshold=0.0001):
        customer_vector = tfidf_vectorizer.transform([customer_data])
        similarity_scores = cosine_similarity(customer_vector, tfidf_matrix).flatten()
        relevant_indices = np.where(similarity_scores > threshold)[0]
        sorted_indices = relevant_indices[np.argsort(similarity_scores[relevant_indices])[::-1]]
        sorted_scores = similarity_scores[sorted_indices]
        skill_names = df_skill.iloc[sorted_indices]['Skill'].values
        return skill_names, sorted_scores

    # Function to cluster data
    def cluster_data(df, n_clusters=2):
        try:
            X = df[['Similarity Score']]
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            df['Cluster'] = kmeans.fit_predict(X)
            cluster_0 = df[df['Cluster'] == 0]
            cluster_1 = df[df['Cluster'] == 1]
            return cluster_0, cluster_1
        except Exception as e:
            return df, df

    # Function to select cluster with higher similarity score
    def max_cluster(cluster_0_df, cluster_1_df):
        if cluster_0_df['Similarity Score'].max() > cluster_1_df['Similarity Score'].max():
            return cluster_0_df
        else:
            return cluster_1_df

    # Function to get skills from greater cluster
    def get_greater_cluster_list(greater_cluster):
        greater_cluster_list = greater_cluster['Skill'].tolist()
        return greater_cluster_list

    # Function to update DataFrame row with skills
    def update_new_df_row(new_df, i, List_greater_cluster):
        for ni in range(1, new_df.shape[1]):
            new_df.iloc[i, ni] = 1 if new_df.columns[ni] in List_greater_cluster else 0
        return new_df

    # Function to process job descriptions and create skill matrix
    def process_job_descriptions(df_job, df_skill, tfidf_matrix, tfidf_vectorizer, threshold=0.0001):
        skills = df_skill['Skill'].tolist()
        new_df = df_job[['Job Title']].copy()
        for skill in skills:
            new_df[skill] = 0
        for job_description, i in zip(df_job.iloc[:, -1], range(new_df.shape[0])):
            top_skills, scores = recommend(job_description, tfidf_matrix, tfidf_vectorizer, df_skill, threshold)
            recommended_skills_df = pd.DataFrame({
                'Skill': top_skills,
                'Similarity Score': scores
            })
            cluster_0_df, cluster_1_df = cluster_data(recommended_skills_df)
            greater_cluster = max_cluster(cluster_0_df, cluster_1_df)
            List_greater_cluster = get_greater_cluster_list(greater_cluster)
            update_new_df_row(new_df, i, List_greater_cluster)
        job_skill_matrix = new_df.copy()
        return job_skill_matrix

    # Vectorize skills and get recommended skills for customer
    tfidf_matrix, tfidf_vectorizer = vectorize_tfidf(df_skill)
    top_skills, scores = recommend(CustomerData, tfidf_matrix, tfidf_vectorizer, df_skill)
    recommended_skills_df = pd.DataFrame({
        'Skill': top_skills,
        'Similarity Score': scores
    })

    # Cluster recommended skills
    cluster_0_df, cluster_1_df = cluster_data(recommended_skills_df)
    greater_cluster = max_cluster(cluster_0_df, cluster_1_df)
    List_greater_cluster = get_greater_cluster_list(greater_cluster)

    # Process job descriptions to create skill matrix
    job_skill_matrix = process_job_descriptions(df_job, df_skill, tfidf_matrix, tfidf_vectorizer, threshold=0.0001)

    # Get jobwant row from skill matrix
    jobwant_row = job_skill_matrix[job_skill_matrix['Job Title'] == Jobwant]

    # Create customer skill DataFrame
    skills = df_skill['Skill'].tolist()
    customer_skill_df = pd.DataFrame(columns=["Job Title"] + skills)
    customer_skill_df.loc[0] = [Jobwant] + [0] * len(skills)
    customer_skill_df = update_new_df_row(customer_skill_df, 0, List_greater_cluster)

    # Merge customer skills with job skills
    job_skill_matrix = job_skill_matrix.loc[:, ~job_skill_matrix.columns.duplicated()]
    customer_skill_df['Job Title'] = customer_skill_df['Job Title'] + " " + CustomerID
    job_skill_similarity = pd.concat([jobwant_row, customer_skill_df], ignore_index=True)

    # Calculate missing skills
    row1 = job_skill_similarity.iloc[0, 1:]
    row2 = job_skill_similarity.iloc[1, 1:]
    difference = row1 - row2
    Miss_skill = difference[difference == 1].index.tolist()

    return Miss_skill
