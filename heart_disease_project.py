import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

@st.cache_data
def load_default_data():
    """Load the default heart.csv dataset."""
    try:
        df = pd.read_csv('heart.csv')
        return df
    except Exception as e:
        st.error(f"Error loading default data: {str(e)}")
        return None

def load_data(file):
    """Load and validate the uploaded dataset."""
    try:
        df = pd.read_csv(file)
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

# [Previous functions remain the same: perform_eda, create_visualizations, 
# perform_time_series_analysis, perform_clustering]

def main():
    st.title("Advanced Data Analysis Application")
    
    st.write("""
    This application provides comprehensive data analysis capabilities including:
    - Exploratory Data Analysis (EDA)
    - Statistical Analysis
    - Data Visualization
    - Time Series Analysis
    - Clustering Analysis
    """)
    
    # Add data source selection
    data_source = st.radio(
        "Choose your data source:",
        ("Use default Heart Disease dataset", "Upload your own dataset")
    )
    
    df = None
    
    if data_source == "Use default Heart Disease dataset":
        df = load_default_data()
        if df is not None:
            st.success("Default Heart Disease dataset loaded successfully!")
    else:
        uploaded_file = st.file_uploader("Upload your dataset (CSV format)", type=['csv'])
        if uploaded_file is not None:
            df = load_data(uploaded_file)
    
    if df is not None:
        # Show raw data
        if st.checkbox("Show raw data"):
            st.write(df)
        
        # Create tabs for different analyses
        tab1, tab2, tab3, tab4 = st.tabs(["EDA", "Visualization", "Time Series", "Clustering"])
        
        with tab1:
            perform_eda(df)
        
        with tab2:
            create_visualizations(df)
        
        with tab3:
            # Convert date columns if present
            date_columns = st.multiselect(
                "Select date columns:",
                df.columns,
                key="date_cols"
            )
            if date_columns:
                for col in date_columns:
                    df[col] = pd.to_datetime(df[col])
                perform_time_series_analysis(df)
            else:
                st.info("Please select date columns for time series analysis")
        
        with tab4:
            perform_clustering(df)

if __name__ == "__main__":
    main()
