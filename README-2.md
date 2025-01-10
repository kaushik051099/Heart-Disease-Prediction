Problem

The project focuses on analyzing a dataset of health metrics (e.g., age, blood pressure, heart rate) from heart patients to develop a predictive model for heart disease diagnosis. The critical goal is to prioritize high recall for identifying all potential heart disease cases, given the severe implications of false negatives.

Objectives

Data Exploration & EDA: Identify patterns and relationships within the data.
Preprocessing:
Remove irrelevant features
Handle missing values, outliers, and skewed distributions
Encode categorical variables and scale features

Model Building:
Create pipelines for scaling and preprocessing
Develop and tune models (KNN, SVM, Decision Tree, Random Forest)
Focus on optimizing recall for positive heart disease cases
Model Evaluation: Compare models using precision, recall, and F1-score.
Steps

EDA: Conducted univariate and bivariate analyses to understand relationships between features and the target variable.
Preprocessing: Addressed data quality issues, including missing values, outliers, and skewness, and prepared the dataset with encoding and scaling.

Model Building:
 - Decision Tree: Baseline and tuned versions evaluated.
 - Random Forest: Enhanced feature aggregation and evaluated performance.
 - KNN: Tuned neighbor count and distance metrics for accuracy.
 - SVM: Implemented with hyperparameter tuning to optimize recall.

Conclusion

The SVM model achieved the highest recall (0.97) for identifying heart disease cases, ensuring comprehensive detection without significantly compromising precision. This balanced performance makes it a reliable choice for medical diagnostics, minimizing false negatives while avoiding excessive false positives.
