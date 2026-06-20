# fraud_detection.py

# ==============================
# 1. Import Libraries
# ==============================
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score
)

from imblearn.over_sampling import SMOTE

# ==============================
# 2. Load Dataset
# ==============================
print("Loading Dataset...\n")

df = pd.read_csv("creditcard.csv")

print("Dataset Loaded Successfully!")
print("Shape:", df.shape)

# ==============================
# 3. Basic Information
# ==============================
print("\nFirst 5 Rows:")
print(df.head())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nClass Distribution:")
print(df["Class"].value_counts())

# ==============================
# 4. Features and Target
# ==============================
X = df.drop("Class", axis=1)
y = df["Class"]

# ==============================
# 5. Train-Test Split
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)

# ==============================
# 6. Apply SMOTE
# ==============================
print("\nApplying SMOTE...")

smote = SMOTE(random_state=42)

X_train_smote, y_train_smote = smote.fit_resample(
    X_train,
    y_train
)

print("\nClass Distribution After SMOTE:")
print(y_train_smote.value_counts())

# ==============================
# 7. Logistic Regression
# ==============================
print("\n================================")
print("LOGISTIC REGRESSION")
print("================================")

lr_model = LogisticRegression(max_iter=1000)

lr_model.fit(X_train_smote, y_train_smote)

y_pred_lr = lr_model.predict(X_test)
y_prob_lr = lr_model.predict_proba(X_test)[:, 1]

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_lr))

print("\nClassification Report:")
print(classification_report(y_test, y_pred_lr))

roc_lr = roc_auc_score(y_test, y_prob_lr)
print("ROC-AUC Score:", round(roc_lr, 4))

# ==============================
# 8. Random Forest
# ==============================
print("\n================================")
print("RANDOM FOREST")
print("================================")

rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train_smote, y_train_smote)

y_pred_rf = rf_model.predict(X_test)
y_prob_rf = rf_model.predict_proba(X_test)[:, 1]

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_rf))

print("\nClassification Report:")
print(classification_report(y_test, y_pred_rf))

roc_rf = roc_auc_score(y_test, y_prob_rf)
print("ROC-AUC Score:", round(roc_rf, 4))

# ==============================
# 9. Compare Models
# ==============================
print("\n================================")
print("MODEL COMPARISON")
print("================================")

print(f"Logistic Regression ROC-AUC : {roc_lr:.4f}")
print(f"Random Forest ROC-AUC       : {roc_rf:.4f}")

if roc_rf > roc_lr:
    print("\nBest Model: Random Forest")
else:
    print("\nBest Model: Logistic Regression")

# ==============================
# 10. Feature Importance
# ==============================
print("\nTop Important Features:")

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf_model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print(feature_importance.head(10))

print("\nProject Completed Successfully!")