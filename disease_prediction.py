import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def load_medical_dataset():
    """
    Loads clinical medical data containing real diagnostic features
    (cell nuclei measurements, radius, texture, area, smoothness, etc.).
    """
    print("[1/4] Loading medical diagnostic dataset...")
    data = load_breast_cancer(as_frame=True)
    df = data.frame

    # Rename target to indicate diagnostic outcome (1 = Benign/Healthy, 0 = Malignant/Disease)
    df = df.rename(columns={"target": "diagnosis"})
    
    print(f"Loaded {df.shape[0]} patient records with {df.shape[1] - 1} clinical features.")
    return df, data.target_names


def evaluate_clinical_model(name: str, model, X_test, y_test, target_names):
    """
    Evaluates classification performance with emphasis on clinical sensitivity and ROC-AUC.
    """
    y_pred = model.predict(X_test)

    # Calculate ROC-AUC if probability outputs or decision margins are supported
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_proba)
    elif hasattr(model, "decision_function"):
        y_scores = model.decision_function(X_test)
        roc_auc = roc_auc_score(y_test, y_scores)
    else:
        roc_auc = None

    print(f"\n{'='*20} {name} Results {'='*20}")
    if roc_auc is not None:
        print(f"ROC-AUC Score: {roc_auc:.4f}")
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nDetailed Clinical Classification Report:")
    print(classification_report(y_test, y_pred, target_names=target_names))


def main():
    # Step 1: Load and prepare data
    df, target_names = load_medical_dataset()

    X = df.drop(columns=["diagnosis"])
    y = df["diagnosis"]

    # Step 2: Stratified train/test split (80% training, 20% testing)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # Step 3: Feature Scaling (Crucial for distance/margin-based models like SVM and Logistic Regression)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("\n[2/4] Initializing diagnostic classifiers...")

    # Classifier 1: Logistic Regression
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train_scaled, y_train)

    # Classifier 2: Support Vector Machine (SVM)
    svm = SVC(kernel="rbf", probability=True, random_state=42)
    svm.fit(X_train_scaled, y_train)

    # Classifier 3: Random Forest Classifier
    rf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
    rf.fit(X_train, y_train)

    # Step 4: Model Evaluation
    print("\n[3/4] Evaluating models against test cohort...")
    evaluate_clinical_model("Logistic Regression", lr, X_test_scaled, y_test, target_names)
    evaluate_clinical_model("Support Vector Machine (SVM)", svm, X_test_scaled, y_test, target_names)
    evaluate_clinical_model("Random Forest Classifier", rf, X_test, y_test, target_names)

    # Top Clinical Biomarkers from Random Forest
    print("\n[4/4] Top Driving Diagnostic Features (Random Forest Importance):")
    importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    for feat, imp in importances.head(8).items():
        print(f" - {feat:<28}: {imp:.4f}")

    # Patient Single Prediction Test
    patient_idx = 0
    sample_patient = X_test_scaled[patient_idx].reshape(1, -1)
    prediction = svm.predict(sample_patient)[0]
    prob = svm.predict_proba(sample_patient)[0][prediction]
    
    print(f"\n--- Diagnostic Simulation for Patient #{patient_idx} ---")
    print(f"Predicted Diagnosis : {target_names[prediction]} (Confidence: {prob * 100:.2f}%)")
    print(f"Actual Diagnosis    : {target_names[y_test.iloc[patient_idx]]}")


if __name__ == "__main__":
    main()