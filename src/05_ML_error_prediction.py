# the goal of this notebook is to use 4 machine learning models:
# logistic regression, Knn, random forest, Svm 
# to predict deepface race-classification errors and compare their performance the baseline classifier

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, classification_report, f1_score, 
                             precision_score, recall_score, roc_auc_score, confusion_matrix)

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


def train_ml_baseline():
    
    BASE_PATH = Path(__file__).parent.parent
    
    plt.rcParams["figure.dpi"] = 120
    
    train = pd.read_parquet(BASE_PATH / "data/ml_ready/train_ml_ready.parquet")
    val   = pd.read_parquet(BASE_PATH / "data/ml_ready/val_ml_ready.parquet")
    
    print(train.shape, val.shape)
    print(train.head())
    # loading the feature datasets i built previously so i can train ML models on them
    
    train["race_true_clean"] = train["race_true"].str.lower().str.replace("_", " ")
    train["pred_race_clean"] = train["pred_race"].str.lower()
    
    val["race_true_clean"] = val["race_true"].str.lower().str.replace("_", " ")
    val["pred_race_clean"] = val["pred_race"].str.lower()
    
    y_train = (train["race_true_clean"] != train["pred_race_clean"]).astype(int)
    y_val   = (val["race_true_clean"]   != val["pred_race_clean"]).astype(int)
    
    feat_cols = ["pred_race_score", "pred_gender_score", "brightness", "contrast", "saturation"]
    
    X_train = train[feat_cols]
    X_val   = val[feat_cols]
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled   = scaler.transform(X_val)
    # cleaning the labels and building the target 0 = correct, 1 = error
    # then selecting and scaling the numeric features for all future ML models
    
    # Nb: to make sure accuracy and the other metrics actually reflect the model's ability to detect deepface errors 
    # i defined the ML target here as 1 = error and 0 = correct.
    # notebook 3 used the opposite convention but only for descriptive analysis
    
    def evaluate_model(y_true, y_pred):
        """
        I realised the dataset is imbalanced class 1 dominates , accuracy alone becomes misleading
        a model can reach ~66% accuracy just by always predicting the majority class
        
        so I add use Precision, Recall, F1-score and ROC-AUC to see if the model actually learns
        to detect real deepface errors instead of just copying the class distribution.
        """
        print("Accuracy:", accuracy_score(y_true, y_pred))
        print("F1-score:", f1_score(y_true, y_pred))
        print("Precision:", precision_score(y_true, y_pred))
        print("Recall:", recall_score(y_true, y_pred))
        print("ROC-AUC:", roc_auc_score(y_true, y_pred))
        print("\nClassification Report:\n", classification_report(y_true, y_pred))
    
    # logistic regression:
    
    log_model = LogisticRegression(max_iter=300, random_state=42)
    log_model.fit(X_train_scaled, y_train)
    log_pred = log_model.predict(X_val_scaled)
    log_acc = accuracy_score(y_val, log_pred)
    print("Logistic Regression:")
    evaluate_model(y_val, log_pred)
    
    cm = confusion_matrix(y_val, log_pred)
    plt.figure(figsize=(4,3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["pred 0 (correct)", "pred 1 (error)"],
                yticklabels=["true 0 (correct)", "true 1 (error)"])
    plt.title("confusion matrix: Logistic Regression")
    plt.show()
    
    # regression performs poorly on this task: 
    # even if accuracy is ~0.667, this number is misleading because the model simply
    # predicts the majority class error = 1 almost all the time
    
    # recall for class 1 is 1.00 it finds every error
    # but recall for class 0 is 0.01 so it totally fails to detect correct predictions
    
    # the model does not learn the real structure of the data it collapses to the majority class
    # in conclusion: logistic regression is too simple for modeling deepface error patterns
    
    # interpretation of the regression:
    # the model predicts the majority class "error" = 1 almost all the time
    # this is why accuracy = 0.667 and recall for class 1 = 1.00
    
    # logistic regression fails to identify the minority class correct = 0
    # only 4 samples out of 698 correct predictions are detected 
    # this confirms that logistic regression is too simple for this task
    
    # KNN:
    
    k_values = [3, 5, 7]
    knn_results = {}
    knn_preds = {}
    
    for k in k_values:
        print(f"\nKNN(k={k})")
        
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train_scaled, y_train)
    
        pred = knn.predict(X_val_scaled)
        knn_results[k] = accuracy_score(y_val, pred)
        knn_preds[k] = pred
    
        print("Accuracy:", knn_results[k])
        evaluate_model(y_val, pred)
        
        cm = confusion_matrix(y_val, pred)
        plt.figure(figsize=(4,3))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["pred 0 (correct)", "pred 1 (error)"],
                    yticklabels=["true 0 (correct)", "true 1 (error)"])
        plt.title(f"confusion matrix : KNN (k={k})")
        plt.show()
    
    print("\nclass balance val:", np.bincount(y_val))
    print("\nclass balance val:", np.bincount(y_val))
    
    # across k = 3, 5, and 7, KNN reaches accuracies 58–60% that remain below the majority-class baseline 66.76% 
    # this shows that KNN does not learn meaningful predictive structure with the current features
    # the confusion matrices show that KNN struggles especially on the minority
    # this provides further evidence that more flexible models are required to capture DeepFace error patterns
    
    # random forest: 
    
    print("\nRandom Forest")
    
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=42,
        class_weight=None
    )
    
    rf_model.fit(X_train_scaled, y_train)
    rf_pred = rf_model.predict(X_val_scaled)
    
    evaluate_model(y_val, rf_pred)
    
    cm = confusion_matrix(y_val, rf_pred)
    
    plt.figure(figsize=(4,3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["pred 0 (correct)", "pred 1 (error)"],
                yticklabels=["true 0 (correct)", "true 1 (error)"])
    plt.title("confusion matrix: Random Forest")
    plt.show()
    
    # random forest : 
    # accuracy = 0.676 is only slightly above the majority-class baseline 0.667
    # the dataset is imbalanced 67% errors, so accuracy alone is misleading
    
    # F1-scores show the real behavior:
    # class 1 error: F1 = 0.79 -> the model is very good at predicting errors
    # class 0 correct: F1 = 0.31 -> it struggles to detect correct predictions
    
    # so random forest learns some non-linear patterns not like logistic regression or Knn 
    # but it still mainly predicts the majority class error and misses most correct cases
    
    # confusion matrix:
    # it correctly catches 1268/1402 errors but only 152/698 correct predictions features are still too weak to model failures
    
    # support vector machine -> SVM:
    
    svm_linear = SVC(kernel="linear", random_state=42)
    svm_linear.fit(X_train_scaled, y_train)
    svm_linear_pred = svm_linear.predict(X_val_scaled)
    
    print("SVM (linear) accuracy:", accuracy_score(y_val, svm_linear_pred))
    evaluate_model(y_val, svm_linear_pred)
    
    cm_linear = confusion_matrix(y_val, svm_linear_pred)
    plt.figure(figsize=(4,3))
    sns.heatmap(cm_linear, annot=True, fmt="d", cmap="Blues",
                xticklabels=["pred 0 (correct)", "pred 1 (error)"],
                yticklabels=["true 0 (correct)", "true 1 (error)"])
    plt.title("confusion matrix: SVM (linear)")
    plt.show()
    
    svm_rbf = SVC(kernel="rbf", random_state=42)
    svm_rbf.fit(X_train_scaled, y_train)
    svm_rbf_pred = svm_rbf.predict(X_val_scaled)
    
    print("SVM (RBF) accuracy:", accuracy_score(y_val, svm_rbf_pred))
    evaluate_model(y_val, svm_rbf_pred)
    
    
    cm_rbf = confusion_matrix(y_val, svm_rbf_pred)
    plt.figure(figsize=(4,3))
    sns.heatmap(cm_rbf, annot=True, fmt="d", cmap="Blues",
                xticklabels=["pred 0 (correct)", "pred 1 (error)"],
                yticklabels=["true 0 (correct)", "true 1 (error)"])
    plt.title("confusion matrix: SVM (RBF)")
    plt.show()
    
    
    print("class balance val:", np.bincount(y_val))
    
    # both SVM models reach accuracies close to the majority baseline 66.76%
    # which confirms that accuracy alone is misleading on this imbalanced dataset
    
    # linear SVM collapses completely : it predicts class 1 error for every sample
    # just like logistic regression meaning it learns no useful decision
    
    # RBF SVM captures more non-linear structure but the improvement is minimal:
    # it still heavily favors the majority class and struggles to detect correct predictions
    # this confirms that the current feature set does not contain strong predictive signals
    
    naive_predictions = np.ones_like(y_val)
    # baseline: always predict "error" majority class = 1
    
    def collect_metrics(y_true, y_pred):
        return {
            "Accuracy": accuracy_score(y_true, y_pred),
            "F1-score": f1_score(y_true, y_pred),
            "Precision": precision_score(y_true, y_pred),
            "Recall": recall_score(y_true, y_pred),
            "ROC-AUC": roc_auc_score(y_true, y_pred)
        }
    
    results = {
        "Logistic Regression": collect_metrics(y_val, log_pred),
        "KNN (k=3)":            collect_metrics(y_val, knn_preds[3]),
        "KNN (k=5)":            collect_metrics(y_val, knn_preds[5]),
        "KNN (k=7)":            collect_metrics(y_val, knn_preds[7]),
        "Random Forest":        collect_metrics(y_val, rf_pred),
        "SVM (linear)":         collect_metrics(y_val, svm_linear_pred),
        "SVM (RBF)":            collect_metrics(y_val, svm_rbf_pred),
        "Baseline (always error)": collect_metrics(y_val, naive_predictions)
    }
    
    df_results = pd.DataFrame(results).T
    df_results = df_results.round(4)
    
    print(df_results)
    
    plt.figure(figsize=(4,3))
    sns.barplot(x=["correct (0)", "error (1)"], y=np.bincount(y_val))
    plt.title("class distribution: validation set")
    plt.ylabel("Count")
    plt.show()
    
    df_plot = df_results.reset_index().rename(columns={"index": "Model"})
    
    plt.figure(figsize=(8,4))
    sns.barplot(data=df_plot, x="Model", y="F1-score")
    plt.xticks(rotation=45, ha="right")
    plt.title("F1-score per model")
    plt.tight_layout()
    plt.show()
    
    plt.figure(figsize=(8,4))
    sns.barplot(data=df_plot, x="Model", y="Accuracy")
    plt.xticks(rotation=45, ha="right")
    plt.title("accuracy per model")
    plt.tight_layout()
    plt.show()
    
    # final conclusion: 
    # unfortunately across all four models performance stays close to the majority-class baseline
    # KNN performs the worst, while Random Forest is the only model that slightly do better
    # all models struggle to detect the minority class correct predictions 
    # and ROC-AUC scores around 0.50 basically means random guessing: these features don't help predict errors at all
    
    # this shows that the available features do not contain strong predictive signals about when deepface will fail...
    # I'll try to achieve higher performance with richer feature representations by updating previous notebooks 
    # Nb: I'll do notebooks BIS so we can compare the before and after


if __name__ == "__main__":
    train_ml_baseline()
