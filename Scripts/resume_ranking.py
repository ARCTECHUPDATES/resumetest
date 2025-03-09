import os
import spacy
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load spaCy NLP model
nlp = spacy.load("en_core_web_sm")

def preprocess_text(text):
    """ Clean and preprocess text using spaCy """
    doc = nlp(text.lower())  
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    return " ".join(tokens)

def load_text_file(filepath):
    """ Load a text file and return its content """
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as file:
            return file.read()
    return ""

def calculate_similarity(job_desc, resumes):
    """ Calculate Cosine Similarity between Job Description and Resumes """
    vectorizer = TfidfVectorizer()
    documents = [job_desc] + resumes  # First document is job description
    tfidf_matrix = vectorizer.fit_transform(documents)
    
    # Compute cosine similarity with job description (first entry)
    similarity_scores = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1:])[0]
    
    return similarity_scores

if __name__ == "__main__":
    # Load job description
    job_desc = preprocess_text(load_text_file("data/job_description.txt"))

    # Load resumes
    resume_files = [f for f in os.listdir("data") if f.endswith(".txt") and "job_description" not in f]
    resumes = [preprocess_text(load_text_file(f"data/{file}")) for file in resume_files]

    # Calculate similarity
    scores = calculate_similarity(job_desc, resumes)

    # Sort resumes by highest similarity score
    ranked_resumes = sorted(zip(resume_files, scores), key=lambda x: x[1], reverse=True)

    # Display results
    df = pd.DataFrame(ranked_resumes, columns=["Resume", "Similarity Score"])
    print(df)