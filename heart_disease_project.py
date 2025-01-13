import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib
from scipy.stats import boxcox

def load_and_prepare_data():
    """Load and prepare the heart disease dataset."""
    df = pd.read_csv('heart.csv')
    
    # One-hot encoding for categorical variables
    df_encoded = pd.get_dummies(df, columns=['cp', 'restecg', 'thal'], drop_first=True)
    
    # Convert other categorical variables to integer
    features_to_convert = ['sex', 'fbs', 'exang', 'slope', 'ca', 'target']
    for feature in features_to_convert:
        df_encoded[feature] = df_encoded[feature].astype(int)
    
    return df_encoded

def train_model(df):
    """Train the Random Forest model with optimized parameters."""
    X = df.drop('target', axis=1)
    y = df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Use the optimized parameters from our analysis
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=3,
        min_samples_split=2,
        min_samples_leaf=1,
        bootstrap=True,
        random_state=42
    )
    
    rf_model.fit(X_train, y_train)
    return rf_model, X_train.columns

def create_prediction_interface():
    """Create the main prediction interface."""
    st.title("Heart Disease Risk Assessment")
    st.write("This application helps assess the risk of heart disease based on various health parameters.")
    
    # Create tabs for different sections
    tab1, tab2, tab3 = st.tabs(["Risk Assessment", "Patient Information", "About Heart Disease"])
    
    with tab1:
        st.subheader("Enter Patient's Medical Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.number_input("Age", min_value=20, max_value=100, value=50)
            sex = st.selectbox("Sex", ["Male", "Female"])
            cp = st.selectbox("Chest Pain Type", 
                            ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"])
            trestbps = st.number_input("Resting Blood Pressure (mm Hg)", 90, 200, 120)
            chol = st.number_input("Cholesterol (mg/dl)", 100, 600, 200)
            
        with col2:
            fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"])
            restecg = st.selectbox("Resting ECG Results", 
                                 ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"])
            thalach = st.number_input("Maximum Heart Rate", 60, 220, 150)
            exang = st.selectbox("Exercise Induced Angina", ["No", "Yes"])
            oldpeak = st.number_input("ST Depression", 0.0, 6.0, 0.0, 0.1)
            
        col3, col4 = st.columns(2)
        with col3:
            slope = st.selectbox("Slope of Peak Exercise ST Segment", 
                               ["Upsloping", "Flat", "Downsloping"])
            ca = st.selectbox("Number of Major Vessels", ["0", "1", "2", "3"])
            
        with col4:
            thal = st.selectbox("Thalassemia", ["Normal", "Fixed Defect", "Reversible Defect"])
        
        if st.button("Assess Risk", type="primary"):
            # Prepare input data
            sex = 1 if sex == "Male" else 0
            fbs = 1 if fbs == "Yes" else 0
            exang = 1 if exang == "Yes" else 0
            
            # Convert categorical variables to match training data format
            cp_dict = {"Typical Angina": 0, "Atypical Angina": 1, 
                      "Non-anginal Pain": 2, "Asymptomatic": 3}
            restecg_dict = {"Normal": 0, "ST-T Wave Abnormality": 1, 
                           "Left Ventricular Hypertrophy": 2}
            slope_dict = {"Upsloping": 0, "Flat": 1, "Downsloping": 2}
            thal_dict = {"Normal": 1, "Fixed Defect": 2, "Reversible Defect": 3}
            
            # Create feature array
            features = [age, sex, cp_dict[cp], trestbps, chol, fbs, 
                       restecg_dict[restecg], thalach, exang, oldpeak, 
                       slope_dict[slope], int(ca), thal_dict[thal]]
            
            # Make prediction
            prediction = model.predict([features])[0]
            probability = model.predict_proba([features])[0][1]
            
            # Display result with custom styling
            st.markdown("---")
            st.subheader("Assessment Result")
            
            col1, col2 = st.columns(2)
            with col1:
                if prediction == 1:
                    st.error("⚠️ Higher Risk of Heart Disease")
                else:
                    st.success("✅ Lower Risk of Heart Disease")
            
            with col2:
                st.metric("Risk Probability", f"{probability:.1%}")
            
            # Show risk factors and recommendations
            st.markdown("### Key Risk Factors")
            risk_factors = []
            if age > 60:
                risk_factors.append("Age above 60")
            if trestbps > 140:
                risk_factors.append("High blood pressure")
            if chol > 200:
                risk_factors.append("High cholesterol")
            if thalach < 100:
                risk_factors.append("Low maximum heart rate")
            
            if risk_factors:
                for factor in risk_factors:
                    st.warning(factor)
                    
                st.info("📋 Recommendation: Please consult with a healthcare provider for a thorough evaluation.")
            
    with tab2:
        st.subheader("Understanding Your Input Parameters")
        st.markdown("""
        - **Age**: Patient's age in years
        - **Blood Pressure**: Resting blood pressure in mm Hg
        - **Cholesterol**: Serum cholesterol in mg/dl
        - **Maximum Heart Rate**: Maximum heart rate achieved during exercise
        - **ST Depression**: ST depression induced by exercise relative to rest
        - **Number of Major Vessels**: Number of major vessels colored by fluoroscopy
        - **Chest Pain Type**: Type of chest pain experienced
        - **Thalassemia**: Blood disorder affecting oxygen-carrying capacity
        """)
        
    with tab3:
        st.subheader("About Heart Disease")
        st.markdown("""
        Heart disease is the leading cause of death globally. Key risk factors include:
        
        1. **High Blood Pressure**: Consistently elevated blood pressure can damage arteries
        2. **High Cholesterol**: Can lead to buildup of plaque in arteries
        3. **Smoking**: Damages blood vessels and reduces oxygen delivery
        4. **Physical Inactivity**: Increases risk of obesity and related conditions
        5. **Excessive Alcohol**: Can lead to high blood pressure and heart muscle damage
        
        Early detection and lifestyle modifications can significantly reduce risk.
        """)
        
        st.warning("⚠️ This tool is for educational purposes only and should not replace professional medical advice.")

# Initialize the app
st.set_page_config(page_title="Heart Disease Risk Assessment", layout="wide")

# Custom CSS
st.markdown("""
    <style>
    .stButton>button {
        width: 100%;
        margin-top: 20px;
    }
    .stAlert {
        padding: 10px;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Load and prepare data
df = load_and_prepare_data()

# Train model
model, feature_names = train_model(df)

# Create the prediction interface
create_prediction_interface()
