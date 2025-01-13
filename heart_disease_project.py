import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

def load_and_prepare_data():
    """Load and prepare the heart disease dataset."""
    try:
        # Load the heart disease dataset
        url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/heart.csv"
        df = pd.read_csv(url)
        
        # Separate features and target
        X = df.drop('target', axis=1)
        y = df['target']
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale the features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        # Train the model
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train_scaled, y_train)
        
        return model, scaler, list(X.columns)
    
    except Exception as e:
        st.error(f"Error in data preparation: {str(e)}")
        st.stop()

def create_prediction_interface(model, scaler, feature_names):
    """Create the streamlit interface for heart disease prediction."""
    st.title("Heart Disease Prediction System")
    
    st.write("""
    This application predicts the likelihood of heart disease based on various medical indicators.
    Please fill in the following information:
    """)
    
    # Create input fields for all features
    features = {}
    
    col1, col2 = st.columns(2)
    
    with col1:
        features['age'] = st.number_input('Age', min_value=20, max_value=100, value=50)
        features['sex'] = st.selectbox('Sex', ['Male', 'Female'])
        features['cp'] = st.selectbox('Chest Pain Type', 
            ['Typical Angina', 'Atypical Angina', 'Non-anginal Pain', 'Asymptomatic'])
        features['trestbps'] = st.number_input('Resting Blood Pressure (mm Hg)', 
            min_value=90, max_value=200, value=120)
        features['chol'] = st.number_input('Cholesterol (mg/dl)', 
            min_value=100, max_value=600, value=200)
        features['fbs'] = st.selectbox('Fasting Blood Sugar > 120 mg/dl', ['No', 'Yes'])
        features['restecg'] = st.selectbox('Resting ECG Results', 
            ['Normal', 'ST-T Wave Abnormality', 'Left Ventricular Hypertrophy'])
    
    with col2:
        features['thalach'] = st.number_input('Maximum Heart Rate', 
            min_value=60, max_value=220, value=150)
        features['exang'] = st.selectbox('Exercise Induced Angina', ['No', 'Yes'])
        features['oldpeak'] = st.number_input('ST Depression Induced by Exercise', 
            min_value=0.0, max_value=6.0, value=0.0, step=0.1)
        features['slope'] = st.selectbox('Slope of Peak Exercise ST Segment', 
            ['Upsloping', 'Flat', 'Downsloping'])
        features['ca'] = st.number_input('Number of Major Vessels Colored by Fluoroscopy', 
            min_value=0, max_value=4, value=0)
        features['thal'] = st.selectbox('Thalassemia', 
            ['Normal', 'Fixed Defect', 'Reversible Defect'])

    # Convert categorical inputs to numerical values
    feature_vector = []
    
    # Age
    feature_vector.append(features['age'])
    
    # Sex
    feature_vector.append(1 if features['sex'] == 'Male' else 0)
    
    # Chest Pain Type
    cp_values = {
        'Typical Angina': 0,
        'Atypical Angina': 1,
        'Non-anginal Pain': 2,
        'Asymptomatic': 3
    }
    feature_vector.append(cp_values[features['cp']])
    
    # Blood Pressure
    feature_vector.append(features['trestbps'])
    
    # Cholesterol
    feature_vector.append(features['chol'])
    
    # Fasting Blood Sugar
    feature_vector.append(1 if features['fbs'] == 'Yes' else 0)
    
    # Resting ECG
    restecg_values = {
        'Normal': 0,
        'ST-T Wave Abnormality': 1,
        'Left Ventricular Hypertrophy': 2
    }
    feature_vector.append(restecg_values[features['restecg']])
    
    # Maximum Heart Rate
    feature_vector.append(features['thalach'])
    
    # Exercise Induced Angina
    feature_vector.append(1 if features['exang'] == 'Yes' else 0)
    
    # ST Depression
    feature_vector.append(features['oldpeak'])
    
    # Slope
    slope_values = {
        'Upsloping': 0,
        'Flat': 1,
        'Downsloping': 2
    }
    feature_vector.append(slope_values[features['slope']])
    
    # Number of Vessels
    feature_vector.append(features['ca'])
    
    # Thalassemia
    thal_values = {
        'Normal': 1,
        'Fixed Defect': 2,
        'Reversible Defect': 3
    }
    feature_vector.append(thal_values[features['thal']])
    
    if st.button('Predict'):
        try:
            # Scale the features
            scaled_features = scaler.transform([feature_vector])
            
            # Make prediction
            prediction = model.predict(scaled_features)[0]
            probability = model.predict_proba(scaled_features)[0][1]
            
            # Display prediction
            st.write('---')
            if prediction == 1:
                st.error(f'⚠️ Heart Disease Detected (Probability: {probability:.2%})')
            else:
                st.success(f'✅ No Heart Disease Detected (Probability: {probability:.2%})')
            
            # Display feature importance
            st.write('### Feature Importance')
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            st.bar_chart(importance_df.set_index('Feature'))
            
        except Exception as e:
            st.error(f"Error in prediction: {str(e)}")

def main():
    """Main function to run the Streamlit app."""
    try:
        # Load model and scaler
        model, scaler, feature_names = load_and_prepare_data()
        
        # Create the prediction interface
        create_prediction_interface(model, scaler, feature_names)
        
    except Exception as e:
        st.error(f"Application error: {str(e)}")

if __name__ == "__main__":
    main()
