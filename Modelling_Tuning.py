import warnings
import mlflow
import mlflow.sklearn
import dagshub
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score,
    confusion_matrix, classification_report
)

warnings.filterwarnings("ignore")

# =========================================================
# SET MLFLOW + DAGSHUB
# =========================================================

dagshub.init(repo_owner='lisaapLI', repo_name='ModellingExspLilisApr', mlflow=True)
mlflow.set_experiment("Diabetes_Classification_Tuning")

# =========================================================
# LOAD DATASET
# =========================================================

df = pd.read_csv('diabetes_clean.csv')

print("Ukuran Data :", df.shape)
print(df.head())

# =========================================================
# FEATURE DAN TARGET
# =========================================================

X = df.drop('Outcome', axis=1)
y = df['Outcome']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =========================================================
# HYPERPARAMETER TUNING
# =========================================================

param_grid = {
    'n_estimators'     : [50, 100, 200],
    'max_depth'        : [None, 5, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf' : [1, 2, 4]
}

grid_search = GridSearchCV(
    estimator  = RandomForestClassifier(random_state=42),
    param_grid = param_grid,
    cv         = 5,
    scoring    = 'accuracy',
    n_jobs     = -1,
    verbose    = 1
)

print("\nMencari hyperparameter terbaik...")
grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
best_model  = grid_search.best_estimator_

print("\nBest Parameters:", best_params)

# =========================================================
# MANUAL LOGGING KE MLFLOW
# =========================================================

with mlflow.start_run(run_name="RandomForest_Diabetes_Tuning"):

    # Prediksi
    y_pred = best_model.predict(X_test)

    # Metrics
    accuracy  = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall    = recall_score(y_test, y_pred)
    f1        = f1_score(y_test, y_pred)

    # ---- Log Parameters ----
    mlflow.log_param("n_estimators",      best_params['n_estimators'])
    mlflow.log_param("max_depth",         best_params['max_depth'])
    mlflow.log_param("min_samples_split", best_params['min_samples_split'])
    mlflow.log_param("min_samples_leaf",  best_params['min_samples_leaf'])
    mlflow.log_param("cv",                5)
    mlflow.log_param("scoring",           "accuracy")

    # ---- Log Metrics ----
    mlflow.log_metric("accuracy",      accuracy)
    mlflow.log_metric("precision",     precision)
    mlflow.log_metric("recall",        recall)
    mlflow.log_metric("f1_score",      f1)
    mlflow.log_metric("best_cv_score", grid_search.best_score_)

    # ---- Log Model ----
    mlflow.sklearn.log_model(
        sk_model      = best_model,
        artifact_path = "modelling_tuning",
        input_example = X_test.iloc[:5]
    )

    # ---- Artifact 1: Confusion Matrix PNG ----
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=['Non-Diabetes', 'Diabetes'],
        yticklabels=['Non-Diabetes', 'Diabetes'],
        ax=ax
    )
    ax.set_title('Confusion Matrix - Random Forest Tuning')
    ax.set_ylabel('Actual')
    ax.set_xlabel('Predicted')
    plt.tight_layout()
    plt.savefig("training_confusion_matrix.png")
    plt.close()
    mlflow.log_artifact("training_confusion_matrix.png")

    # ---- Artifact 2: Classification Report ----
    report = classification_report(y_test, y_pred)
    with open("classification_report.txt", "w") as f:
        f.write(report)
    mlflow.log_artifact("classification_report.txt")

    # ---- Artifact 3: Feature Importance ----
    feat_imp = pd.Series(
        best_model.feature_importances_,
        index=X.columns
    ).sort_values(ascending=False)

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    feat_imp.plot(kind='bar', ax=ax2, color='steelblue')
    ax2.set_title('Feature Importance - Random Forest Tuning')
    ax2.set_ylabel('Importance Score')
    plt.tight_layout()
    plt.savefig("feature_importance.png")
    plt.close()
    mlflow.log_artifact("feature_importance.png")

    # ---- Print Hasil ----
    print("\n=== HASIL EVALUASI ===")
    print("Accuracy :", accuracy)
    print("Precision:", precision)
    print("Recall   :", recall)
    print("F1 Score :", f1)
    print("Best CV Score:", grid_search.best_score_)

    print("\n=== CONFUSION MATRIX ===")
    print(cm)

    print("\n=== CLASSIFICATION REPORT ===")
    print(report)