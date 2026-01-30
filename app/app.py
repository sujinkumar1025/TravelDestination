import os
import base64
import joblib
import streamlit as st
import pandas as pd
import numpy as np


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# -----------------------------
# Load model and columns
# -----------------------------
model = joblib.load(os.path.join(BASE_DIR, "models", "travel_model2.pkl"))
model_columns = joblib.load(os.path.join(BASE_DIR, "models", "model_columns2.pkl"))
# model = joblib.load("models/travel_model2.pkl")
# model_columns = joblib.load("models/model_columns2.pkl")

def get_base64_image(img_path):
    with open(img_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
img_path = os.path.join(BASE_DIR, "asssets", "4299.jpg")

bg_img = get_base64_image(img_path)

css = f"""
<style>
/* App container */
[data-testid="stAppViewContainer"] {{
    position: relative;
    z-index: 1;
}}

/* Blurred background layer */
[data-testid="stAppViewContainer"]::before {{
    content: "";
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background-image: url("data:image/jpg;base64,{bg_img}");
    background-size: cover;
    background-position: center;
    filter: blur(18px);
    transform: scale(1.1); /* prevents edge blur cutoff */
    z-index: -1;
}}

/* Transparent header */
[data-testid="stHeader"] {{
    background: transparent;
}}
</style>
"""

st.markdown(css, unsafe_allow_html=True)

st.title("🌍 Travel Destination Prediction")
st.write("Enter traveler preferences to predict the best destination:")

# -----------------------------
# User Inputs
# -----------------------------
budget_options = ["Low","Medium","High"]
climate_options = ["Cold","Warm","Hot"]
travel_type_options = ["Beach","Adventure","Historical"]
season_options = ["Summer","Winter","Monsoon"]
crowd_options = ["Low","Medium","High"]
accommodation_options = ["Hotel","Resort","Homestay","Lodge"]
family_options = ["Yes","No"]
travel_method_options = ["Bus","Train","Flight","Car"]

col1, col2 = st.columns(2)

with col1:
    budget = st.selectbox("Budget", budget_options)
    climate = st.selectbox("Climate", climate_options)
    travel_type = st.selectbox("Travel Type", travel_type_options)
    best_season = st.selectbox("Best Season", season_options)
    duration_days = st.slider("Duration (days)", 3, 20, 5)
    rating = st.slider("Rating (2.0–5.0)", 2.0, 5.0, 3.0, step=0.1)

with col2:
    avg_temperature = st.number_input(
        "Average Temperature (°C)", min_value=-10, max_value=50, value=25
    )
    avg_cost = st.number_input(
        "Average Cost (₹)", min_value=1000, max_value=500000, value=20000
    )
    crowd_level = st.selectbox("Crowd Level", crowd_options)
    accommodation_type = st.selectbox("Accommodation Type", accommodation_options)
    family_friendly = st.selectbox("Family Friendly", family_options)
    travel_method = st.selectbox("Travel Method", travel_method_options)


# -----------------------------
# Prepare input DataFrame
# -----------------------------
input_dict = {
    'Budget': budget,
    'Climate': climate,
    'Travel_Type': travel_type,
    'Best_Season': best_season,
    'Duration_Days': duration_days,
    'Avg_Temperature': avg_temperature,
    'Avg_Cost': avg_cost,
    'Rating': rating,
    'Crowd_Level': crowd_level,
    'Accommodation_Type': accommodation_type,
    'Family_Friendly': family_friendly,
    'Travel_Method': travel_method
}

input_df = pd.DataFrame([input_dict])

# One-hot encode input to match training columns
input_encoded = pd.get_dummies(input_df)
for col in model_columns:
    if col not in input_encoded.columns:
        input_encoded[col] = 0
input_encoded = input_encoded[model_columns]  # ensure correct column order

# -----------------------------
# Prediction
# -----------------------------
# if st.button("Predict Destination"):
#     prediction = model.predict(input_encoded)[0]
#     st.success(f"🏖️ Recommended Travel Destination: {prediction}")
def destination_card(name, probability):
    img_path = os.path.join(BASE_DIR, "asssets","destinations", f"{name.lower()}.jpg")

    if not os.path.exists(img_path):
        img_path = os.path.join(BASE_DIR, "asssets", "default.jpg")

    with st.container():
        st.image(img_path, use_container_width=True)
        st.markdown(f"### 🌍 {name}")
        st.progress(int(probability * 100))
        # st.caption(f"Confidence: {probability:.2%}")


# if st.button("Predict Destination"):
#     # Get probabilities
#     probabilities = model.predict_proba(input_encoded)[0]
#     destinations = model.classes_

#     # Get top 3 indices
#     top3_idx = np.argsort(probabilities)[-3:][::-1]

#     st.success("🏖️ Top 3 Recommended Travel Destinations:")

#     for i, idx in enumerate(top3_idx, start=1):
#         st.write(f"{i}. **{destinations[idx]}** — {probabilities[idx]:.2f}")

if st.button("Predict Destination"):
    probabilities = model.predict_proba(input_encoded)[0]
    destinations = model.classes_

    top3_idx = np.argsort(probabilities)[-3:][::-1]

    st.subheader("✨ Top 3 Travel Recommendations")

    cols = st.columns(3)

    for col, idx in zip(cols, top3_idx):
        with col:
            destination_card(destinations[idx], probabilities[idx])
