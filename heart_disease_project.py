import os

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
    default_file = "https://github.com/kaushik051099/Heart-Disease-Prediction/blob/e6115404f65287341ce603519891408b55e412e0/heart.csv"  # Specify the default dataset file name
    
    if uploaded_file is not None:
        # Load data from uploaded file
        df = load_data(uploaded_file)
    elif os.path.exists(default_file):
        st.info(f"No file uploaded. Using default dataset: {default_file}")
        df = load_data(default_file)
    else:
        st.error("No file uploaded, and the default dataset is missing. Please upload a dataset.")
        return
    
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
