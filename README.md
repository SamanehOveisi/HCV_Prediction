
HCV Prediction Using Machine Learning

This repository contains a project designed to predict Hepatitis C Virus (HCV) conditions using machine learning models. The primary focus is to classify patients into various medical categories using data analysis, feature selection, and predictive modelling.
----------------------------------

Project Features: 
Random Forest Classifier:
Used for initial predictions and performance evaluation.
Employs cross-validation to measure accuracy and reliability.
Logistic Regression with Feature Selection:
Balances the dataset using SMOTE (Synthetic Minority Over-sampling Technique).
Selects optimal features using Recursive Feature Elimination with Cross-Validation (RFECV).
Ensemble Learning:
Combines predictions from Random Forest and Logistic Regression using an Artificial Bee Colony (ABC) optimization algorithm.
Visualization:
Visualizes class distribution, accuracy across folds, confusion matrices, and more.
---------------------------------
Requirements:
To run the project, ensure the following dependencies are installed:

pandas
numpy
scikit-learn
imblearn
matplotlib
seaborn
openpyxl
You can install the required libraries with:
pip install -r requirements.txt
---------------------------------
Workflow:

Data Loading:
The dataset HCV_modified.xlsx is read and processed for analysis.
Data Preprocessing:
Normalize numerical features using MinMaxScaler.
Encode categorical features (e.g., Sex) with LabelEncoder.
Map target classes to integers for compatibility with models.
Model Training and Evaluation:
Train Random Forest and Logistic Regression models using cross-validation.
Measure model performance using accuracy, confusion matrices, and classification reports.
SMOTE and Feature Selection:
Apply SMOTE to balance imbalanced classes.
Use RFECV for feature selection in Logistic Regression.
Ensemble Modeling:
Blend model predictions using the ABC algorithm for optimal performance.
----------------------------------
How to Run:

1. Clone the repository:
git clone https://github.com/yourusername/HCV_Prediction.git
cd HCV_Prediction

2. Ensure the dataset file HCV_modified.xlsx is placed in the correct directory.

3. Run the main script:
python main.py

4. The program outputs:
Accuracy and metrics in the console.
Visualizations of class distribution, accuracy, and confusion matrices.
----------------------------------
Key Outputs:

Average Accuracy:
Random Forest: ~X% (update with actual results).
Logistic Regression: ~X% (update with actual results).
Ensemble Model: ~X% (update with actual results).
Visualizations:
Class distribution bar chart.
Accuracy plots across folds.
Confusion matrix heatmaps
----------------------------------
Directory Structure:

HCV_Prediction/
├── HCV_modified.xlsx       # Dataset
├── main.py                 # Main script for training and evaluation
├── requirements.txt        # Dependencies
├── README.md               # Project documentation
-----------------------------------
Future Work:

Experiment with additional models like XGBoost or LightGBM.
Implement hyperparameter tuning for Random Forest and Logistic Regression.
Explore additional ensemble techniques for improved predictions.
-----------------------------------
License:

This project is licensed under the MIT License. See the LICENSE file for details.
------------------------------------
Authors:

Samaneh Oveisi, Maryam Farhangi, Mahmoud Nikbakht

For any inquiries, contact S.oveisikahkha@student.fdu.edu
------------------------------------
Enjoy exploring the project and predicting HCV conditions! 🚀
