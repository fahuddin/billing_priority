import pandas as pd 
from ..config import DATA_PATH
import nltk
import spacy
from nltk.sentiment.vader import SentimentIntensityAnalyzer
nlp = spacy.load("en_core_web_sm")
def load_raw_data():
    df = pd.read_csv(DATA_PATH)
    return df 

def preprocess_text(text):
    """Clean and lemmatize text using SpaCy."""
    if pd.isna(text):
        return ""
    doc = nlp(str(text).lower())
    tokens = [token.lemma_ for token in doc if not token.is_punct and not token.is_stop]
    return " ".join(tokens)

#scores based on text
sid = SentimentIntensityAnalyzer()
def Sentiment_Analyzer(column: pd.Series):
    return column.apply(lambda x: sid.polarity_scores(str(x))['compound'] if pd.notna(x) else 0)


def extract_entities(text):
    """Extract medical entities using SpaCy NER."""
    if pd.isna(text):
        return []
    doc = nlp(str(text))
    return [ent.text.lower() for ent in doc.ents if ent.label_ in ['DISEASE', 'SYMPTOM', 'MEDICATION']]


def count_urgent_keywords(text):
    urgent_keywords = [
    'pain', 'fever', 'breath', 'sore throat', 'vomiting', 'diarrhea', 'chest', 
    'emergency', 'severe', 'chronic', 'malignant', 'sepsis', 'fracture'
]
    """Count occurrences of urgent keywords in text."""
    if pd.isna(text):
        return 0
    text = str(text).lower()
    return sum(text.count(keyword) for keyword in urgent_keywords)

def map_icd_severity(column: pd.Series):
    icd_severity = {
    'F32.9': 4,   # Major depressive disorder, single episode, unspecified → Severe
    'E11.9': 3,   # Type 2 diabetes mellitus without complications → Moderate
    'I10': 3,     # Essential (primary) hypertension → Moderate
    'K21.9': 2,   # GERD without esophagitis → Mild to Moderate
    'Z00.00': 1,  # General medical exam without complaint → None/Mild
    'L40.0': 2,   # Psoriasis vulgaris → Mild to Moderate (can vary)
    'J45.909': 3, # Unspecified asthma, uncomplicated → Moderate (if controlled)
    'N18.3': 4,   # Chronic kidney disease stage 3 → Moderate to Severe
    'R51': 1      # Headache → Mild (unless chronic or underlying issue)
}
    return column.map(map_icd_severity).fillna(1)

def column_length(column:pd.Series):
    return column.apply(lambda x: x.fillna('').str.split().str.len())

