import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
# --- New Import: We will use a more powerful model ---
from sklearn.ensemble import RandomForestRegressor
import streamlit as st
import base64 # Import to handle custom styling

# --- Data Loading and Preprocessing (Keep as is) ---
medical_df = pd.read_csv('insurance.csv')

# Encoding maps for the model
medical_df.replace({'sex':{'male':0,'female':1}},inplace=True)
medical_df.replace({'smoker':{'yes':0,'no':1}},inplace=True)
medical_df.replace({'region':{'southeast':0,'southwest':1,'northwest':2,'northeast':3}},inplace=True)

X= medical_df.drop('charges',axis=1)
y = medical_df['charges']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=2)

# --- MODEL SELECTION ---
# We are changing the model from LinearRegression to RandomForestRegressor
# Old line: lg = LinearRegression()
new_model = RandomForestRegressor(n_estimators=100, random_state=42)
new_model.fit(X_train,y_train)

# --- Streamlit Web App (Enhanced Frontend with Table Layout) ---

st.set_page_config(page_title="Insurance Charges Predictor", layout="centered")

st.title("🏥 Medical Insurance Charges Predictor")
st.markdown("Enter the following details to estimate the medical insurance charge.")

# Create a container for the input table-like structure
st.header("Patient Data")

# --- Layout Setup (Mimics a 2-Row, 3-Column Table) ---

# First Row of Features: Age, Sex, BMI
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Age")
    # Number Input for Age
    age = st.number_input("Age", min_value=18, max_value=100, value=30, step=1, label_visibility="collapsed")

with col2:
    st.subheader("Gender")
    # Radio for Sex (male=0, female=1) - Looks like buttons
    sex = st.radio(
        "Gender",
        ('Male', 'Female'),
        index=0,
        horizontal=True,
        label_visibility="collapsed"
    )
    sex_map = {'Male': 0, 'Female': 1}

with col3:
    st.subheader("BMI")
    # Number Input for BMI
    bmi = st.number_input("BMI (e.g., 27.9)", min_value=10.0, max_value=50.0, value=25.0, step=0.1, format="%.1f", label_visibility="collapsed")

# Separator for visual clarity
st.markdown("---")

# Second Row of Features: Children, Smoker, Region
col4, col5, col6 = st.columns(3)

with col4:
    st.subheader("Children")
    # Slider for Children
    children = st.slider("Children", min_value=0, max_value=5, value=0, label_visibility="collapsed")

with col5:
    st.subheader("Smoker")
    # Radio for Smoker (yes=0, no=1) - Looks like buttons
    smoker = st.radio(
        "Smoker?",
        ('No', 'Yes'),
        index=0,
        horizontal=True,
        label_visibility="collapsed"
    )
    smoker_map = {'Yes': 0, 'No': 1}
    
with col6:
    st.subheader("Region")
    # Selectbox for Region (or use Radio if you prefer 4 buttons)
    region = st.selectbox(
        "Region",
        ('Northeast', 'Northwest', 'Southeast', 'Southwest'),
        index=0,
        label_visibility="collapsed"
    )
    region_map = {'Southeast': 0, 'Southwest': 1, 'Northwest': 2, 'Northeast': 3}


# --- Prediction Logic ---

# Convert user inputs to the model's expected format (numerical array)
user_input_data = [
    age,
    sex_map[sex],
    bmi,
    children,
    smoker_map[smoker],
    region_map[region]
]

st.markdown("---")

# The 'Predict' button is placed centrally
predict_col = st.columns([1, 2, 1])[1]
with predict_col:
    if st.button('💰 Estimate Charges', use_container_width=True):
        
        # Reshape the data for the model
        final_features = np.array(user_input_data).reshape(1, -1)
        
        # Make the prediction using the new model
        prediction = new_model.predict(final_features)
        
        # Display the result
        st.success("### Estimated Medical Insurance Charge:")
        # Format the charge as currency
        st.balloons()
        st.write(f"## **${prediction[0]:,.2f}**")
        # --- Update the caption to reflect the new model ---
        st.caption("This estimation is based on the trained Random Forest model.")
