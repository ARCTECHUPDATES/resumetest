import streamlit as st
import os
import spacy
import pandas as pd
import subprocess
import pdfplumber
import pytesseract
import cv2
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image

# Ensure spaCy model is installed
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"], check=True)
    nlp = spacy.load("en_core_web_sm")

# Function to extract text from PDF (including OCR for scanned PDFs)
def extract_text_from_pdf(pdf_file):
    """ Extract text from a PDF file, including scanned PDFs using OCR """
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            extracted_text = page.extract_text()
            if extracted_text:
                text += extracted_text + "\n"
            else:
                # If text extraction fails, use OCR
                image = page.to_image(resolution=300)
                text += pytesseract.image_to_string(image.original) + "\n"
    return text.strip()

# Function to extract text from images using OCR
def extract_text_from_image(image_file):
    """ Extract text from an image using OCR """
    image = Image.open(image_file).convert("RGB")
    text = pytesseract.image_to_string(image)
    return text.strip()

# Text preprocessing using spaCy (lemmatization & stopword removal)
def preprocess_text(text):
    """ Preprocess text using spaCy (tokenization, lemmatization, stopwords removal) """
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    return " ".join(tokens)

# Function to calculate similarity between job description and resumes
def calculate_similarity(job_desc, resumes):
    """ Calculate Cosine Similarity between Job Description and Resumes """
    vectorizer = TfidfVectorizer()
    documents = [job_desc] + resumes
    tfidf_matrix = vectorizer.fit_transform(documents)
    similarity_scores = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1:])[0]
    return similarity_scores

# Streamlit UI
st.title("🚀 AI-Powered Resume Screening & Ranking System")

# Upload Job Description
st.header("📜 Upload Job Description")
job_desc_file = st.file_uploader("Choose a Job Description file (.txt)", type=["txt"])

# Upload Resumes (Supports both TXT and PDF)
st.header("📂 Upload Resumes")
resume_files = st.file_uploader("Upload Multiple Resumes (.txt, .pdf)", type=["txt", "pdf"], accept_multiple_files=True)

if job_desc_file and resume_files:
    # Read job description
    job_desc = job_desc_file.read().decode("utf-8")
    job_desc = preprocess_text(job_desc)

    # Read and preprocess resumes
    resumes = []
    resume_names = []

    for resume_file in resume_files:
        if resume_file.type == "application/pdf":
            resume_text = extract_text_from_pdf(resume_file)  # Extract text from PDF
        else:
            resume_text = resume_file.read().decode("utf-8")  # Extract text from TXT

        resumes.append(preprocess_text(resume_text))
        resume_names.append(resume_file.name)

    # Calculate similarity scores
    scores = calculate_similarity(job_desc, resumes)

    # Convert similarity scores to percentage
    scores_percentage = [round(score * 100, 2) for score in scores]

    # Sort resumes by similarity score
    ranked_resumes = sorted(zip(resume_names, scores_percentage), key=lambda x: x[1], reverse=True)

    # Display results
    df = pd.DataFrame(ranked_resumes, columns=["Resume", "Candidate Score (%)"])
    st.subheader("📊 Ranked Resumes")
    st.dataframe(df)
