import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

# Function to calculate feature weights (unchanged)
def calculate_feature_weights(df, target):
    features = [col for col in df.select_dtypes(include=['number']).columns if col != target]
    X = df[features]
    y = df[target]
    
    if y.isna().sum() > 0:
        valid_indices = y.notna()
        X = X[valid_indices]
        y = y[valid_indices]
    
    if X.isna().any().any():
        X = X.fillna(X.mean())
    
    model = RandomForestRegressor(random_state=42)
    model.fit(X, y)
    
    importances = dict(zip(features, model.feature_importances_))
    total = sum(importances.values())
    weights = {f: imp / total for f, imp in importances.items()}
    return weights, features

# Function to filter drugs by condition (unchanged)
def filter_drugs_by_condition(df, condition):
    return df[df['medical_condition'].str.contains(condition, case=False, na=False)]

# Function to score a drug (unchanged)
def score_drug(row, weights, features, invert_features=None):
    if invert_features is None:
        invert_features = []
    score = 0.0
    for feature in features:
        if feature in weights:
            if feature in invert_features:
                score += weights[feature] * (1 - row[feature])
            else:
                score += weights[feature] * row[feature]
    return score

# Modified recommend_drugs function to include pregnancy and alcohol filters
def recommend_drugs(df, condition, weights, features, top_n=5, is_pregnant=False, uses_alcohol=False):
    filtered = filter_drugs_by_condition(df, condition).copy()
    if filtered.empty:
        return pd.DataFrame(columns=["drug_name", "score"])
    
    # Apply pregnancy filter: exclude drugs with high-risk pregnancy categories (D, X)
    if is_pregnant:
        filtered = filtered[~filtered['pregnancy_category'].isin(['D', 'X'])]
    
    # Apply alcohol filter: exclude drugs with 'X' in alcohol column
    if uses_alcohol:
        filtered = filtered[filtered['alcohol'] != 'X']
    
    if filtered.empty:
        return pd.DataFrame(columns=["drug_name", "score"])
    
    filtered.loc[:, "score"] = filtered.apply(lambda row: score_drug(row, weights, features), axis=1)
    ranked = filtered.sort_values(by="score", ascending=False)
    return ranked[["drug_name", "score", "pregnancy_category", "alcohol"]].head(top_n)

# Streamlit app
st.title("Personalized Drug Recommendation System")
st.markdown("""
Welcome to the Drug Recommendation System! Select a medical condition and specify your personal conditions (e.g., pregnancy or alcohol use) to see the top recommended drugs tailored to your needs.
""")

# Load the dataset
try:
    df = pd.read_csv("drugsdata.csv")
except FileNotFoundError:
    st.error("Error: 'drugsdata.csv' not found. Please ensure the file is in the same directory as this script.")
    st.stop()

# Calculate feature weights
try:
    weights, features = calculate_feature_weights(df, target="rating")
except Exception as e:
    st.error(f"Error calculating feature weights: {e}")
    st.stop()

# Get unique medical conditions for the dropdown
conditions = sorted(df['medical_condition'].unique())

# Create input fields
st.subheader("Your Preferences")
selected_condition = st.selectbox("Select a Medical Condition", conditions)
is_pregnant = st.checkbox("I am pregnant", value=False)
uses_alcohol = st.checkbox("I consume alcohol", value=False)

# Button to trigger recommendation
if st.button("Get Recommendations"):
    if selected_condition:
        # Get recommendations with filters
        top_drugs = recommend_drugs(
            df, 
            selected_condition, 
            weights, 
            features, 
            top_n=5, 
            is_pregnant=is_pregnant, 
            uses_alcohol=uses_alcohol
        )
        
        if top_drugs.empty:
            st.warning(f"No drugs found for the condition '{selected_condition}' with the specified preferences.")
        else:
            st.subheader(f"Top 5 Recommended Drugs for {selected_condition}")
            # Display results in a table with additional columns
            st.table(top_drugs)
            
            # Detailed recommendations with additional info
            st.markdown("### Detailed Recommendations")
            for idx, row in top_drugs.iterrows():
                st.markdown(
                    f"- **{row['drug_name']}**: Score = {row['score']:.2f}, "
                    f"Pregnancy Category = {row['pregnancy_category']}, "
                    f"Alcohol Interaction = {row['alcohol'] or 'Not specified'}"
                )
    else:
        st.warning("Please select a medical condition.")

# Add footer
st.markdown("""
---
*Built with Streamlit. Data sourced from drugsdata.csv. Recommendations are based on a Random Forest model using feature importance. Pregnancy and alcohol filters applied as needed.*
""")