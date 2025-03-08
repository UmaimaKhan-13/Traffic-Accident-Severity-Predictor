import streamlit as st
import pandas as pd
import joblib
import os

# Define file paths
MODEL_PATH = r"C:\Users\NA\Downloads\Traffic Accident Severity streamlit deployement\xgb_model.pkl"
LABEL_ENCODER_PATH = r"C:\Users\NA\Downloads\Traffic Accident Severity streamlit deployement\label_encoder.pkl"
# Load the trained model
if os.path.exists(MODEL_PATH):
    xgb_model = joblib.load(MODEL_PATH)
else:
    st.error("🚨 Model file not found! Please check the file path.")

# Load the label encoder
try:
    label_encoder = joblib.load(LABEL_ENCODER_PATH)
except FileNotFoundError:
    label_encoder = None

# Define numerical and categorical columns
numerical_columns = ['Time', 'Number_of_vehicles_involved', 'Number_of_casualties']
categorical_columns = ['Day_of_week_Monday', 'Day_of_week_Saturday', 'Day_of_week_Sunday',
                       'Age_band_of_driver_Under 18', 'Age_band_of_driver_Unknown',
                       'Area_accident_occured_Unknown', 'Types_of_Junction_No junction',
                       'Types_of_Junction_Other', 'Light_conditions_Darkness - no lighting',
                       'Weather_conditions_Other', 'Type_of_collision_Collision with pedestrians',
                       'Vehicle_movement_Overtaking']

# Custom CSS for Dark Blue Theme 🎨
st.markdown(
    """
    <style>
        .stApp {
            background-color: #0A192F;
            color: white;
        }
        .title {
            font-size: 38px;
            font-weight: bold;
            text-align: center;
            color: #00BFFF;
            padding: 10px;
            border-radius: 10px;
        }
        .stButton>button {
            background-color: #008CBA;
            color: white;
            border-radius: 10px;
            padding: 10px;
            font-size: 18px;
        }
        .stSidebar {
            background-color: #112240 !important;
            color: white !important;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar with Model Info 🏗️
st.sidebar.title("🛠️ About This App")
st.sidebar.write("This app predicts **Accident Severity** based on input features. 🚦")
st.sidebar.write("🔹 **Model**: XGBoost Classifier")
st.sidebar.write("🔹 **Developer**: Umaima")
st.sidebar.write("🔹 **Category Inputs**: True / False")

# Main App Title (Bold & Styled)
st.markdown('<p class="title">🚦 <b>Accident Severity Prediction</b></p>', unsafe_allow_html=True)

st.markdown("### 📌 Enter details below to predict accident severity:")

# User input fields
user_input = {}

# Input numerical features
st.markdown("#### 🔢 Numerical Features")
for feature in numerical_columns:
    user_input[feature] = st.number_input(f"🔹 {feature}:", min_value=0.0, format="%.2f")

# Input categorical features (Dropdown with only True/False)
st.markdown("#### 🏷️ Categorical Features")
for feature in categorical_columns:
    user_input[feature] = st.selectbox(f"🔹 {feature}:", [True, False])

# Convert user input into DataFrame
user_df = pd.DataFrame([user_input])

# Prediction Button
if st.button("🚀 Predict Severity"):
    if "xgb_model" in locals():
        predicted_encoded = xgb_model.predict(user_df)[0]

        # Decode prediction if label encoder is available
        if label_encoder:
            predicted_decoded = label_encoder.inverse_transform([predicted_encoded])[0]
        else:
            predicted_decoded = predicted_encoded  # If no encoder, keep as is

        st.success(f"✅ **Predicted Severity: {predicted_decoded}**")
    else:
        st.error("⚠️ Model is not loaded. Please check the file path.")
