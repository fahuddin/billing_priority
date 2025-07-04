# 📊 Billing Case Severity Classifier

## Overview

Hospital billing teams often face immense pressure during cost report deadlines, which can sometimes lead to critical billing cases being overlooked. This project aims to mitigate that issue by **classifying billing cases based on severity**, enabling billing teams to **prioritize important cases** efficiently and accurately.

By leveraging machine learning techniques — specifically a **Random Forest Classifier** and **Bidirectional Encoder Representations (BERT)** — this model learns patterns from the case data and predicts the severity level for each billing instance.

---

## 💡 Problem Statement

> **"Can we help hospital billing teams avoid missing critical billing cases by automatically classifying and prioritizing them based on severity?"**

This project addresses the challenge by developing a machine learning pipeline to classify cases and assist the billing team with an intelligent prioritization system.

---

## 🧠 Approach

1. **Data Preprocessing**  
   - Cleaned and encoded structured features such as `age`, `narrative sentiment`, `length of narrative`, `time of day`, etc.
   - Extracted sentence embeddings using a **Bidirectional Encoder (BERT)** model from the unstructured narrative texts.

2. **Modeling**  
   - Combined structured features and BERT embeddings to create a comprehensive feature set.
   - Trained a **Random Forest Classifier** to predict the `priority` or severity label for each case.
   - Applied cross-validation and fine-tuned with **GridSearchCV** to select the best hyperparameters.

3. **Evaluation**  
   - Measured model performance using **F1-score**, **precision**, and **recall** to ensure high accuracy, especially for the critical cases.

---

## 🔧 Technologies Used

- Python 🐍
- scikit-learn
- XGBoost (optional)
- pandas / numpy
- GridSearchCV


## 📂 Project Structure
<pre> 
├── data/ # Processed and raw data
├── models/ # Trained models and checkpoints
├── notebooks/ # Jupyter notebooks for experiments
├── src/ # Source code and ML pipeline
│ ├── preprocess.py
│ ├── train_model.py
│ └── evaluate.py
├── requirements.txt
└── README.md
 </pre>
