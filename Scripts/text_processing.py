import spacy

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

def preprocess_text(text):
    """ Preprocess text using spaCy (tokenization, stopwords removal, lemmatization) """
    doc = nlp(text.lower())  
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    return " ".join(tokens)

# Example Usage
sample_text = "I have 5 years of experience in Python and Data Science."
print(preprocess_text(sample_text))
