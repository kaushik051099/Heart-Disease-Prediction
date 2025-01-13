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

def load_data(file):
    """Load and validate the uploaded dataset."""
    try:
        df = pd.read_csv(file)
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

def perform_eda(df):
    """Perform exploratory data analysis on the dataset."""
    st.subheader("Data Overview")
    
    # Display basic information
    st.write("Dataset Shape:", df.shape)
    st.write("Data Types:")
    st.write(df.dtypes)
    
    # Display summary statistics
    st.subheader("Summary Statistics")
    st.write(df.describe())
    
    # Missing values analysis
    missing_values = df.isnull().sum()
    if missing_values.any():
        st.subheader("Missing Values Analysis")
        st.write(missing_values[missing_values > 0])
        
        fig, ax = plt.subplots(figsize=(10, 6))
        missing_values.plot(kind='bar')
        plt.title("Missing Values by Column")
        plt.xticks(rotation=45)
        st.pyplot(fig)

def create_visualizations(df):
    """Create various visualizations based on the data types."""
    st.subheader("Data Visualization")
    
    # Select columns for visualization
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    
    # Distribution plots for numerical columns
    if len(numeric_cols) > 0:
        st.write("### Distribution Analysis")
        selected_num_col = st.selectbox("Select column for distribution analysis:", numeric_cols)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Histogram
        sns.histplot(data=df, x=selected_num_col, ax=ax1)
        ax1.set_title(f"Histogram of {selected_num_col}")
        
        # Box plot
        sns.boxplot(data=df, y=selected_num_col, ax=ax2)
        ax2.set_title(f"Box Plot of {selected_num_col}")
        
        st.pyplot(fig)
    
    # Correlation analysis
    if len(numeric_cols) > 1:
        st.write("### Correlation Analysis")
        correlation = df[numeric_cols].corr()
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0, ax=ax)
        plt.title("Correlation Heatmap")
        st.pyplot(fig)
    
    # Categorical analysis
    if len(categorical_cols) > 0:
        st.write("### Categorical Analysis")
        selected_cat_col = st.selectbox("Select categorical column:", categorical_cols)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        df[selected_cat_col].value_counts().plot(kind='bar')
        plt.title(f"Distribution of {selected_cat_col}")
        plt.xticks(rotation=45)
        st.pyplot(fig)

def perform_time_series_analysis(df):
    """Perform time series analysis if date column is present."""
    date_cols = df.select_dtypes(include=['datetime64']).columns
    if len(date_cols) > 0:
        st.subheader("Time Series Analysis")
        
        # Select date column and target variable
        date_col = st.selectbox("Select date column:", date_cols)
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        target_col = st.selectbox("Select target variable:", numeric_cols)
        
        # Set date as index
        ts_df = df.set_index(date_col)
        
        # Perform decomposition
        decomposition = seasonal_decompose(ts_df[target_col], period=12)
        
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 10))
        decomposition.observed.plot(ax=ax1)
        ax1.set_title("Observed")
        decomposition.trend.plot(ax=ax2)
        ax2.set_title("Trend")
        decomposition.seasonal.plot(ax=ax3)
        ax3.set_title("Seasonal")
        decomposition.resid.plot(ax=ax4)
        ax4.set_title("Residual")
        plt.tight_layout()
        st.pyplot(fig)
        
        # Perform ADF test
        st.write("### Stationarity Test (ADF)")
        adf_result = adfuller(ts_df[target_col].dropna())
        st.write(f"ADF Statistic: {adf_result[0]}")
        st.write(f"p-value: {adf_result[1]}")

def perform_clustering(df):
    """Perform K-means clustering on numerical data."""
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    if len(numeric_cols) >= 2:
        st.subheader("Clustering Analysis")
        
        # Select features for clustering
        selected_features = st.multiselect("Select features for clustering:", numeric_cols, default=list(numeric_cols)[:2])
        
        if len(selected_features) >= 2:
            # Prepare data
            X = df[selected_features]
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Perform PCA if more than 2 dimensions
            if len(selected_features) > 2:
                pca = PCA(n_components=2)
                X_pca = pca.fit_transform(X_scaled)
            else:
                X_pca = X_scaled
            
            # Perform K-means
            n_clusters = st.slider("Number of clusters:", 2, 10, 3)
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            clusters = kmeans.fit_predict(X_scaled)
            
            # Plot results
            fig, ax = plt.subplots(figsize=(10, 6))
            scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis')
            plt.colorbar(scatter)
            plt.title("K-means Clustering Results")
            if len(selected_features) > 2:
                plt.xlabel("First Principal Component")
                plt.ylabel("Second Principal Component")
            else:
                plt.xlabel(selected_features[0])
                plt.ylabel(selected_features[1])
            st.pyplot(fig)

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
    
    uploaded_file = st.file_uploader("Upload your dataset (CSV format)", type=['csv'])
    
    if uploaded_file is not None:
        # Load data
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
