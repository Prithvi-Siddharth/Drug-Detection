import streamlit as st
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, RobustScaler, PowerTransformer, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
import re

# Streamlit app title
st.title("Drug Recommendation System")

# Load and preprocess data
@st.cache_data
def load_and_preprocess_data():
    df = pd.read_csv('drugsdata.csv')
    
    # Drop unnecessary columns
    df = df.drop(columns=['related_drugs', 'drug_link', 'medical_condition_url', 'brand_names', 
                         'medical_condition_description', 'generic_name'])
    
    # Convert activity to float
    df['activity'] = df['activity'].str.rstrip('%').astype(float) / 100
    
    # Handle missing values
    df['no_of_reviews'] = df['no_of_reviews'].fillna(0)
    df['rating'] = df['rating'].fillna(df['rating'].mean())
    df['alcohol'] = df['alcohol'].fillna('Unknown')
    
    # Count side effects
    def count_side_effects(side_effects_str):
        if pd.isna(side_effects_str) or side_effects_str == '':
            return 0
        effects = re.split(r'[;,.\\n]', str(side_effects_str))
        effects = [effect.strip().lower() for effect in effects if effect.strip()]
        return len(effects)
    
    # Count drug classes
    def count_drug_classes(classes_str):
        if pd.isna(classes_str) or classes_str == '':
            return 0
        classes = [c.strip() for c in classes_str.split(',') if c.strip()]
        return len(classes)
    
    df['num_side_effects'] = df['side_effects'].apply(count_side_effects)
    df['num_drug_classes'] = df['drug_classes'].apply(count_drug_classes)
    
    # Drop original side_effects and drug_classes
    df = df.drop(columns=['side_effects', 'drug_classes'])
    
    # Encode categorical features
    categorical_features = ['rx_otc', 'pregnancy_category', 'csa', 'alcohol']
    le_dict = {}
    for col in categorical_features:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        le_dict[col] = le
    
    # Encode medical condition
    le_condition = LabelEncoder()
    df['encoded_medical_condition'] = le_condition.fit_transform(df['medical_condition'].astype(str))
    
    # Define numerical features and preprocessing pipeline
    numerical_features = ['activity', 'no_of_reviews', 'num_side_effects', 'num_drug_classes']
    numeric_transformer = Pipeline(steps=[
        ('power', PowerTransformer(method='yeo-johnson')),
        ('scaler', RobustScaler())
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_features)
        ], remainder='passthrough'
    )
    
    # Prepare features and target
    all_features = numerical_features + categorical_features + ['encoded_medical_condition']
    X = df[all_features]
    y = df['rating']
    
    # Apply preprocessing
    processed_X = preprocessor.fit_transform(X)
    processed_columns = numerical_features + categorical_features + ['encoded_medical_condition']
    final_df = pd.DataFrame(processed_X, columns=processed_columns, index=df.index)
    final_df['drug_name'] = df['drug_name']
    final_df['medical_condition'] = df['medical_condition']
    final_df['rating'] = y
    
    # Scale numerical features to [0, 1]
    scaler = MinMaxScaler()
    final_df[numerical_features] = scaler.fit_transform(final_df[numerical_features])
    
    # Scale categorical features to [0, 1]
    categorical_to_scale = categorical_features + ['encoded_medical_condition']
    scaler_cat = MinMaxScaler()
    final_df[categorical_to_scale] = scaler_cat.fit_transform(final_df[categorical_to_scale])
    
    return final_df, le_condition

# Calculate feature weights
def calculate_feature_weights(df, target='rating'):
    features = [col for col in df.select_dtypes(include=['number']).columns 
                if col != target and col not in ['drug_name', 'medical_condition']]
    if not features:
        raise ValueError('No numerical features available for model fitting.')
    
    X = df[features]
    y = df[target]
    
    X = X.fillna(X.mean())
    valid_indices = y.notna()
    X = X[valid_indices]
    y = y[valid_indices]
    
    model = RandomForestRegressor(random_state=42)
    model.fit(X, y)
    
    importances = dict(zip(features, model.feature_importances_))
    total = sum(importances.values())
    weights = {f: imp / total for f, imp in importances.items()}
    return weights, features

# Score a drug
def score_drug(row, weights, features, invert_features=None):
    if invert_features is None:
        invert_features = ['num_side_effects']
    score = 0.0
    for feature in features:
        if feature in weights:
            value = row[feature]
            if feature in invert_features:
                score += weights[feature] * (1 - value)
            else:
                score += weights[feature] * value
    return score

# Filter drugs by condition
def filter_drugs_by_condition(df, condition):
    return df[df['medical_condition'] == condition]

# Recommend drugs
def recommend_drugs(df, condition, weights, features, top_n=5):
    filtered = filter_drugs_by_condition(df, condition).copy()
    if filtered.empty:
        return pd.DataFrame(columns=['drug_name', 'score'])
    
    filtered['score'] = filtered.apply(lambda row: score_drug(row, weights, features), axis=1)
    ranked = filtered.sort_values(by='score', ascending=False)
    return ranked[['drug_name', 'score']].head(top_n)

# Load data and calculate weights
final_df, le_condition = load_and_preprocess_data()
weights, features = calculate_feature_weights(final_df)

# Streamlit UI
st.header("Select a Medical Condition")
conditions = sorted(final_df['medical_condition'].unique())
selected_condition = st.selectbox("Choose a condition:", conditions)

# Display recommendations
if selected_condition:
    st.header(f"Top Recommended Drugs for {selected_condition}")
    top_drugs = recommend_drugs(final_df, selected_condition, weights, features, top_n=5)
    if not top_drugs.empty:
        st.table(top_drugs)
    else:
        st.write("No drugs found for this condition.")

# Display feature weights
st.header("Feature Weights")
weights_df = pd.DataFrame(list(weights.items()), columns=['Feature', 'Weight'])
st.table(weights_df)

# Instructions
st.write("""
### How to Use
1. Select a medical condition from the dropdown menu.
2. View the top 5 recommended drugs with their scores.
3. Check the feature weights used to calculate the scores.
""")