import streamlit as st
import os
import spacy
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

def preprocess_text(text):
    """ Preprocess text using spaCy (tokenization, lemmatization, stopwords removal) """
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    return " ".join(tokens)

def calculate_similarity(job_desc, resumes):
    """ Calculate Cosine Similarity between Job Description and Resumes """
    vectorizer = TfidfVectorizer()
    documents = [job_desc] + resumes
    tfidf_matrix = vectorizer.fit_transform(documents)
    
    similarity_scores = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1:])[0]
    return similarity_scores

# Streamlit UI
st.title("📄 Resume Screening & Ranking System")

# Upload Job Description
st.header("Upload Job Description")
job_desc_file = st.file_uploader("Choose a Job Description file (.txt)", type=["txt"])

# Upload Resumes
st.header("Upload Resumes")
resume_files = st.file_uploader("Upload Multiple Resumes (.txt)", type=["txt"], accept_multiple_files=True)

if job_desc_file and resume_files:
    # Read job description
    job_desc = job_desc_file.read().decode("utf-8")
    job_desc = preprocess_text(job_desc)

    # Read and preprocess resumes
    resumes = []
    resume_names = []
    for resume_file in resume_files:
        resume_text = resume_file.read().decode("utf-8")
        resumes.append(preprocess_text(resume_text))
        resume_names.append(resume_file.name)

    # Calculate similarity scores
    scores = calculate_similarity(job_desc, resumes)

    # Sort resumes by similarity score
    ranked_resumes = sorted(zip(resume_names, scores), key=lambda x: x[1], reverse=True)

    # Display results
    df = pd.DataFrame(ranked_resumes, columns=["Resume", "Similarity Score"])
    st.subheader("📊 Ranked Resumes")
    st.dataframe(df)

