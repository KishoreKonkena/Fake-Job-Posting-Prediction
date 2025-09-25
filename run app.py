# -*- coding: utf-8 -*-
"""
Created on Thu Sep 25 09:47:35 2025

@author: kirsh
"""

import pandas as pd
import joblib
import os

save_path = r"C:\Users\kirsh\OneDrive\Desktop\sql and python\Deploy"

# Load objects
model = joblib.load(os.path.join(save_path, "xgb_fake_jobs_model.pkl"))
vectorizer = joblib.load(os.path.join(save_path, "tfidf_vectorizer.pkl"))
ohe_columns = joblib.load(os.path.join(save_path, "ohe_columns.pkl"))
threshold = joblib.load(os.path.join(save_path, "best_threshold.pkl"))

# Example new job posting
new_post = {
    "telecommuting": 0,
    "has_company_logo": 1,
    "has_questions": 0,
    "employment_type": "Full-time",
    "required_experience": "Entry level",
    "required_education": "Bachelor's Degree",
    "combined_text": "We are hiring a data scientist with experience in Python and ML."
}

new_df = pd.DataFrame([new_post])

# --- Step 1: OneHotEncode ---
new_dummies = pd.get_dummies(
    new_df[['telecommuting','has_company_logo','has_questions','employment_type','required_experience','required_education']],
    columns=['telecommuting','has_company_logo','has_questions','employment_type','required_experience','required_education'],
    drop_first=True
).astype(int)

# Align with training columns
new_dummies = new_dummies.reindex(columns=ohe_columns, fill_value=0)

# --- Step 2: TF-IDF ---
tfidf_features = vectorizer.transform(new_df['combined_text'])
tfidf_df = pd.DataFrame(tfidf_features.toarray(), columns=vectorizer.get_feature_names_out())

# --- Step 3: Combine features ---
new_processed = pd.concat([new_dummies, tfidf_df], axis=1)

# --- Step 4: Predict ---
y_proba = model.predict_proba(new_processed)[:, 1]
y_pred = (y_proba >= threshold).astype(int)

print("Predicted class:", y_pred[0])
print("Fraud probability:", y_proba[0])
