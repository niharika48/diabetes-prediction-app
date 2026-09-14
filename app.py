import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# 1. Page Configuration (Makes it look professional)
st.set_page_config(page_title="Diabetes Prediction System", page_icon="🩺", layout="centered")

# Add an attractive header image (you can replace this URL with a local image file if you prefer)
st.image("https://images.unsplash.com/photo-1579684385127-1ef15d508118?q=80&w=1000&auto=format&fit=crop", use_container_width=True)
st.title("🩺 Interactive Diabetes Prediction System")
st.write("Enter the patient's medical details below to predict the likelihood of diabetes using our Support Vector Machine (SVM) model.")

st.markdown("---")

# 2. Cache the model training so it doesn't retrain every time you click a button
@st.cache_resource
def load_and_train_model():
    # Load dataset
    diabetes = pd.read_csv('diabetes.csv')
    
    # Separate features and labels
    X = diabetes.drop('Outcome', axis=1)
    y = diabetes['Outcome']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42, stratify=y)
    
    # Create and train pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svc', SVC(kernel='linear', C=0.1)) # Using C=0.1 as a generally safe default to prevent overfitting
    ])
    pipeline.fit(X_train, y_train)
    
    return pipeline

# Load the trained model
model = load_and_train_model()

# 3. Create the User Interface (Input Fields)
st.subheader("Patient Data Input")

col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input("Number of Pregnancies", min_value=0, max_value=20, value=1)
    glucose = st.number_input("Glucose Level", min_value=0, max_value=200, value=85)
    blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=150, value=66)
    skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=29)

with col2:
    insulin = st.number_input("Insulin Level", min_value=0, max_value=900, value=0)
    bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=26.6)
    dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.351)
    age = st.number_input("Age", min_value=1, max_value=120, value=31)

# 4. Prediction Logic
st.markdown("---")
if st.button("Predict Diabetes Status", type="primary"):
    
    # Format the user inputs into a numpy array
    input_data = (pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age)
    input_data_as_numpy_array = np.asarray(input_data).reshape(1, -1)
    
    # Make the prediction
    prediction = model.predict(input_data_as_numpy_array)
    
    # Display the result attractively
    if prediction[0] == 1:
        st.error("⚠️ **Prediction:** This person is likely to have diabetes. Please consult a doctor.")
    else:
        st.success("✅ **Prediction:** This person is unlikely to have diabetes.")