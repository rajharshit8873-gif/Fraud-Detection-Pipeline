Fraud Detection Pipeline
Overview

This project focuses on detecting fraudulent credit card transactions using supervised machine learning techniques. The dataset contains highly imbalanced classes, where fraudulent transactions represent only a small fraction of the total records. To address this challenge, SMOTE (Synthetic Minority Over-sampling Technique) was applied to balance the dataset and improve model performance.

Objectives
Detect fraudulent credit card transactions with high accuracy and reliability.
Handle class imbalance using SMOTE.
Compare the performance of multiple classification algorithms.
Evaluate models using Precision, Recall, F1-Score, and ROC-AUC instead of relying solely on accuracy.
Technologies Used
Python
Pandas
NumPy
Scikit-Learn
Imbalanced-Learn (SMOTE)
Machine Learning Models
Logistic Regression
Random Forest Classifier
Project Workflow
Data Loading and Exploration
Data Preprocessing
Class Imbalance Handling using SMOTE
Train-Test Split
Model Training
Logistic Regression
Random Forest
Model Evaluation
Feature Importance Analysis
Model Comparison and Selection
Evaluation Metrics
Precision
Recall
F1-Score
ROC-AUC Score
Confusion Matrix
Key Features
Handles highly imbalanced fraud detection data.
Uses SMOTE for synthetic oversampling.
Compares multiple machine learning algorithms.
Identifies important features influencing fraud detection.
Provides performance-driven model evaluation.
Results

The models were successfully trained and evaluated on the fraud detection dataset. Random Forest demonstrated strong performance in identifying fraudulent transactions and achieved superior classification results compared to the baseline model.

Repository Structure
Fraud-Detection-Pipeline/
│
├── fraud_detection.py
├── README.md
├── requirements.txt
└── .gitignore
Author

Data Science Internship Project – Fraud Detection Pipeline
