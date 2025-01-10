import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from scipy.stats import boxcox

# Set page configuration
st.set_page_config(page_title="Heart Disease Analysis", layout="wide")

# Load data
@st.cache
def load_data():
    return pd.read_csv('heart.csv')

df = load_data()

# Sidebar
st.sidebar.header("Options")
show_data = st.sidebar.checkbox("Show Raw Data", value=False)
show_summary = st.sidebar.checkbox("Show Data Summary", value=True)
show_visuals = st.sidebar.checkbox("Show Visualizations", value=True)

# Main title
st.title("Heart Disease Analysis")

# Show raw data
if show_data:
    st.subheader("Raw Data")
    st.write(df)

# Data summary
if show_summary:
    st.subheader("Data Summary")
    st.write(df.describe(include='all').T)

# Univariate visualizations
if show_visuals:
    st.subheader("Visualizations")

    # Continuous features
    continuous_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    categorical_features = [col for col in df.columns if col not in continuous_features + ['target']]

    # Continuous Features Distribution
    st.write("### Continuous Features Distribution")
    fig, ax = plt.subplots(2, 3, figsize=(15, 10))
    for i, col in enumerate(continuous_features):
        sns.histplot(df[col], kde=True, ax=ax[i // 3, i % 3], color='pink')
        ax[i // 3, i % 3].set_title(col)
    st.pyplot(fig)

    # Categorical Features Distribution
    st.write("### Categorical Features Distribution")
    fig, ax = plt.subplots(3, 3, figsize=(15, 15))
    for i, col in enumerate(categorical_features):
        sns.countplot(data=df, x=col, ax=ax[i // 3, i % 3], palette="Set2")
        ax[i // 3, i % 3].set_title(col)
    st.pyplot(fig)

# Preprocessing
st.write("### Preprocessing")
df_encoded = pd.get_dummies(df, columns=['cp', 'restecg', 'thal'], drop_first=True)
for feature in ['sex', 'fbs', 'exang', 'slope', 'ca', 'target']:
    df_encoded[feature] = df_encoded[feature].astype(int)

X = df_encoded.drop('target', axis=1)
y = df_encoded['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Models
st.write("### Model Training and Evaluation")
models = {"Decision Tree": DecisionTreeClassifier(),
          "Random Forest": RandomForestClassifier(),
          "K-Nearest Neighbors": KNeighborsClassifier(),
          "SVM": Pipeline([("scaler", StandardScaler()), ("svm", SVC(probability=True))])}

results = []
for name, model in models.items():
    if name in ["K-Nearest Neighbors", "SVM"]:
        model.fit(X_train, y_train)
    else:
        model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results.append({"Model": name, "Accuracy": acc})

results_df = pd.DataFrame(results).sort_values(by="Accuracy", ascending=False)
st.write(results_df)

# Best Model Visualization
st.bar_chart(results_df.set_index("Model"))

import streamlit as st

# Title and instructions
st.title("Heart Disease Prediction")
st.write("Enter the patient's details to predict the likelihood of heart disease.")

# User input with sliders and labels
age = st.slider("Age (years)", 20, 100, 50)
trestbps = st.slider("Blood Pressure (mm Hg)", 90, 200, 120)
chol = st.slider("Cholesterol Level (mg/dL)", 100, 350, 220)
thalach = st.slider("Maximum Heart Rate (bpm)", 60, 220, 140)
oldpeak = st.slider("ST Depression (oldpeak)", 0.0, 6.0, 1.0)

# Show input summary
st.write(f"Age: {age}")
st.write(f"Blood Pressure: {trestbps} mm Hg")
st.write(f"Cholesterol: {chol} mg/dL")
st.write(f"Max Heart Rate: {thalach} bpm")
st.write(f"Oldpeak: {oldpeak}")

# Prediction button (you can replace with your model's prediction function)
if st.button("Predict"):
    prediction = predict_heart_disease(age, trestbps, chol, thalach, oldpeak)
    st.write(f"Prediction: {prediction}")
    
# Predict button
if st.button("Predict"):
    input_data = np.array([[age, trestbps, chol, thalach, oldpeak]])
    prediction = model.predict(input_data)
    if prediction[0] == 1:
        st.write("Prediction: Likely to have heart disease")
    else:
        st.write("Prediction: Unlikely to have heart disease")
