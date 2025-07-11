from ..data import load_raw_data, column_length, preprocess_text, preprocess_text, map_icd_severity, count_urgent_keywords, Sentiment_Analyzer
from ..features import feature_builder
from .evaluate import evaluate_model
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np



def train():
    df = load_raw_data()
    df = df.drop(columns=['PATIENT ID'])
    df['Date of Visit'] = pd.to_datetime(df['Date of Visit'])
    df['Date of Birth'] = pd.to_datetime(df['Date of Birth'])
    
    df = df.assign(
    narrative_cleaned = df["narrative"].apply(preprocess_text),
    narrative_sentiment = Sentiment_Analyzer(df["narrative"]),
    narrative_length = column_length(df["narrative"]),
    urgent_score = df["narrative"].apply(count_urgent_keywords),
    map_icd_severity = df['Diagnosis Code'].apply(map_icd_severity),
    priority_score = np.where(
    (df['map_icd_severity'] >= 3) |
    (df['urgent_score'] >= 2),1,0))


    model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    model.fit(X_feat, y)

    evaluate_model(model, X_feat, y)
    return model

if __name__ == "__main__":
    train()
