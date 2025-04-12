import re# Import required libraries
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from xulidulieu import Pre  # Assuming this is a custom module
import warnings
warnings.filterwarnings("ignore")


CustomerData = "using python and tableau to processing data"  # Description of current skills
Field = "Data Science"  # Field of interest
Jobwant = "Data Analyst"  # Job title or role you want to apply for
CustomerID = "C00001"
# Model for recommendation: "All" for all outputs, "Matrix" for job skill matrix, "Miss_skill" for missing skills, "Job" for job title
def Recommendation_model(CustomerData, Field, Jobwant, CustomerID,model="Miss_skill"):


    # Load data from Excel files
    df_job = pd.read_excel("Job.xlsx", sheet_name=Field)  # Update with the correct path
    df_skill = pd.read_excel("Skill.xlsx")  # Update with the correct path

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
print(Recommendation_model(CustomerData, Field, Jobwant, CustomerID, model="All"))  # Example usage


