import streamlit as st
import numpy as np
import pandas as pd
import ast
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Load stored embeddings
df = pd.read_csv("advisers_dummy_data.csv")

# Load embedding model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Define weights for similarity scores  
weight_expertise = 0.4
weight_past_thesis = 0.6

# Convert JSON string embeddings to numpy arrays (this handles the stored embeddings as JSON strings)
df["expertise_embeddings"] = df["expertise_embeddings"].apply(ast.literal_eval)
df["past_thesis_embeddings"] = df["past_thesis_embeddings"].apply(ast.literal_eval)

# Convert to numpy arrays for cosine similarity calculation
expertise_embeddings = np.stack(df["expertise_embeddings"].values)
past_thesis_embeddings = np.stack(df["past_thesis_embeddings"].values)

# Page Config
st.set_page_config(
    page_title="Advisify: Thesis Adviser Recommender",
    page_icon="🤵",
    layout="centered",
)

# Title and Instructions
st.title("🎓 Advisify: Thesis Adviser Recommender")
st.markdown("Enter your thesis title")
st.markdown("⚠️ For the most accurate matching, please provide a clear and specific thesis title that describes the focus of your research. Avoid vague titles like 'Thesis about AI.'")

# Thesis Title Input
user_input = st.text_area(
    "Input your thesis title (max 300 characters)", 
    placeholder="Enter a detailed thesis title (e.g., 'Optimizing Machine Learning Models for Predicting Crop Yields in Pampanga')", 
    max_chars=300
)

# Project Types Selection (Optional)
project_types = [
    "Web App", "Mobile App", "Desktop Application", "IoT System",
    "Data Analytics / Data Science", "AI / Machine Learning", "Chatbot / NLP",
    "Game Development", "Augmented Reality / Virtual Reality", "Embedded Systems",
    "Networking / Cybersecurity", "Automation / Robotics", "Information System",
    "Decision Support System", "Expert System", "Recommender System",
    "Blockchain App", "E-commerce System"
]

st.markdown("### Select general project types (optional)")
selected_project_types = st.multiselect(
    "If your thesis title is too vague, select one or more general project types to help refine recommendations:",
    project_types
)

# Recommender System Logic
if st.button("Find Adviser"):
    if user_input.strip() or selected_project_types:
        input_text = user_input.strip()
        if selected_project_types:
            input_text += f" ({', '.join(selected_project_types)})"

        user_embedding = model.encode([input_text])

        # Calculate cosine similarity with the advisers' embeddings
        sim_expertise = cosine_similarity(user_embedding, expertise_embeddings)
        sim_past_thesis = cosine_similarity(user_embedding, past_thesis_embeddings)

        # Combine the similarity scores using weighted average
        overall_similarity = (
            weight_expertise * sim_expertise +
            weight_past_thesis * sim_past_thesis
        )

        # Add similarity results to the dataframe
        df["adviser_area_of_expertise_similarity"] = np.clip(sim_expertise[0], 0, 1)
        df["adviser_past_thesis_supervised_similarity"] = np.clip(sim_past_thesis[0], 0, 1)
        df["overall_similarity"] = np.clip(overall_similarity[0], 0, 1)

        # Get top 5 recommendations based on overall similarity
        top_recommendations = df.nlargest(5, "overall_similarity")

        st.subheader("📌 Recommended Advisers:")
        for _, row in top_recommendations.iterrows():
            st.markdown(f"### 🎓 {row['name']}")
            st.write(f"📖 {row['past_thesis_topics_supervised']}")
            st.write(f"🧠 {row['area_of_expertise_description']}")

            # Display similarity progress bars
            st.progress(row["adviser_area_of_expertise_similarity"])  
            st.write(f"💼 **Adviser's Area of Expertise Fit:** {round(row['adviser_area_of_expertise_similarity'] * 100, 2)}%")

            st.progress(row["adviser_past_thesis_supervised_similarity"])  
            st.write(f"📚 **Adviser's Past Thesis Supervised Fit:** {round(row['adviser_past_thesis_supervised_similarity'] * 100, 2)}%")

            st.progress(row["overall_similarity"])  
            st.write(f"⭐ **Overall Match:** {round(row['overall_similarity'] * 100, 2)}%")

            st.write("---")
    else:
        st.warning("⚠️ Please enter your thesis title or select at least one project type before clicking 'Find Adviser'.")
