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
from docx import Document

# Streamlit Cloud ke liye Tesseract ka path set kiya hai
pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

# Ensure spaCy model is installed
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"], check=True)
    nlp = spacy.load("en_core_web_sm")

# Function to extract text from PDF (including OCR for scanned PDFs)
def extract_text_from_pdf(pdf_file):
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            extracted_text = page.extract_text()
            if extracted_text:
                text += extracted_text + "\n"
            else:
                image = page.to_image(resolution=300)
                text += pytesseract.image_to_string(image.original) + "\n"
    return text.strip()

# Function to extract text from images (JPG, PNG, JPEG) using OCR
def extract_text_from_image(image_file):
    image = Image.open(image_file).convert("RGB")
    return pytesseract.image_to_string(image).strip()

# Function to extract text from Word documents (.docx)
def extract_text_from_docx(docx_file):
    doc = Document(docx_file)
    return "\n".join([para.text for para in doc.paragraphs]).strip()

# Text preprocessing using spaCy (lemmatization & stopword removal)
def preprocess_text(text):
    doc = nlp(text.lower())
    return " ".join([token.lemma_ for token in doc if not token.is_stop and not token.is_punct])

# Function to calculate similarity between job description and resumes
def calculate_similarity(job_desc, resumes):
    vectorizer = TfidfVectorizer()
    documents = [job_desc] + resumes
    tfidf_matrix = vectorizer.fit_transform(documents)
    return cosine_similarity(tfidf_matrix[0], tfidf_matrix[1:])[0]

# Streamlit UI
st.title("🚀 AI-Powered Resume Screening & Ranking System")

# Job Description Input – Choose either Upload OR Paste
st.header("📜 Job Description")
jd_option = st.radio("How do you want to provide the Job Description?", 
                     ("Paste Job Description", "Upload a File"))

job_desc = ""

if jd_option == "Paste Job Description":
    job_desc = st.text_area("Paste Job Description Here")

elif jd_option == "Upload a File":
    job_desc_file = st.file_uploader("Upload a Job Description file (.txt, .pdf, .docx, .jpg, .png, .jpeg)", 
                                     type=["txt", "pdf", "docx", "jpg", "jpeg", "png"])

    if job_desc_file:
        ext = job_desc_file.name.split(".")[-1].lower()
        if ext == "txt":
            job_desc = job_desc_file.read().decode("utf-8")
        elif ext == "pdf":
            job_desc = extract_text_from_pdf(job_desc_file)
        elif ext == "docx":
            job_desc = extract_text_from_docx(job_desc_file)
        elif ext in ["jpg", "jpeg", "png"]:
            job_desc = extract_text_from_image(job_desc_file)

# Preprocess Job Description for better matching
if job_desc:
    job_desc = preprocess_text(job_desc)

# Upload Resumes (Supports TXT, PDF, JPG, PNG, JPEG, DOCX)
st.header("📂 Upload Resumes")
resume_files = st.file_uploader("Upload Multiple Resumes (.txt, .pdf, .jpg, .png, .jpeg, .docx)", 
                                type=["txt", "pdf", "jpg", "jpeg", "png", "docx"], accept_multiple_files=True)

if job_desc and resume_files:
    resumes = []
    resume_names = []

    for resume_file in resume_files:
        ext = resume_file.name.split(".")[-1].lower()
        if ext == "txt":
            resume_text = resume_file.read().decode("utf-8")
        elif ext == "pdf":
            resume_text = extract_text_from_pdf(resume_file)
        elif ext == "docx":
            resume_text = extract_text_from_docx(resume_file)
        elif ext in ["jpg", "jpeg", "png"]:
            resume_text = extract_text_from_image(resume_file)
        
        resumes.append(preprocess_text(resume_text))
        resume_names.append(resume_file.name)

    # Calculate similarity scores
    scores = calculate_similarity(job_desc, resumes)
    scores_percentage = [round(score * 100, 2) for score in scores]

    # Sort resumes by similarity score
    ranked_resumes = sorted(zip(resume_names, scores_percentage), key=lambda x: x[1], reverse=True)

    # Display results
    df = pd.DataFrame(ranked_resumes, columns=["Resume", "Candidate Score (%)"])
    st.subheader("📊 Ranked Resumes")
    st.dataframe(df)
