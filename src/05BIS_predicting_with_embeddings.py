# goal of this notebook:
# train several machine learning models to predict when deepface makes race classification errors better than my first attempt
# using the new dataset that includes both my initial ML features and the embeddings
# i hope this will give better result than in notebook 05 

import os
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, classification_report,
    f1_score, precision_score, recall_score, roc_auc_score,
    confusion_matrix
)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import xgboost as xgb

import seaborn as sns
import matplotlib.pyplot as plt


def train_ml_with_embeddings():
    
    BASE_PATH = Path(__file__).parent.parent
    
    train_path = BASE_PATH / "data" / "ml_final" / "train_ml_final.parquet"
    val_path   = BASE_PATH / "data" / "ml_final" / "val_ml_final.parquet"
    
    print(train_path)
    print(val_path)
    # setting up imports and loading the finals datasets
    
    train = pd.read_parquet(train_path)
    val   = pd.read_parquet(val_path)
    
    print(train.shape, val.shape)
    # loading ml_final datasets created in 04BIS
    
    train["race_true_clean"] = train["race_true"].str.lower().str.replace("_", " ")
    train["pred_race_clean"] = train["pred_race"].str.lower()
    
    val["race_true_clean"] = val["race_true"].str.lower().str.replace("_", " ")
    val["pred_race_clean"] = val["pred_race"].str.lower()
    
    y_train = (train["race_true_clean"] != train["pred_race_clean"]).astype(int)
    y_val   = (val["race_true_clean"]   != val["pred_race_clean"]).astype(int)
    
    print("class balance train:", np.bincount(y_train))
    print("class balance val:", np.bincount(y_val))
    # target = 1 when deepface is wrong nd 0 otherwise
    
    X_train = train.select_dtypes(include=[np.number])
    X_val   = val.select_dtypes(include=[np.number])
    
    print("train feature shape:", X_train.shape)
    print("val feature shape:", X_val.shape)
    # isolating the real ML features: the embeddings nd the numeric scores so the models train on actual signals
    
    # Fill NaN with 0 instead of removing rows (to keep all embeddings)
    print(f"NaN count before fill: train={X_train.isna().sum().sum()}, val={X_val.isna().sum().sum()}")
    X_train = X_train.fillna(0).values  # convert to numpy
    X_val = X_val.fillna(0).values
    print(f"after fillna: train shape={X_train.shape}, val shape={X_val.shape}")
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    # scaling numeric features for models
    
    def evaluate_model(y_true, y_pred):
        print("accuracy:", accuracy_score(y_true, y_pred))
        print("f1-score:", f1_score(y_true, y_pred))
        print("precision:", precision_score(y_true, y_pred))
        print("recall:", recall_score(y_true, y_pred))
        
        try:
            print("roc-auc:", roc_auc_score(y_true, y_pred))
        except:
            print("roc-auc: undefined (model predicted only one class)")
        
        print("\nclassification report:\n", classification_report(y_true, y_pred))
     # same than in nb 05 but i added for safety that ROC-AUC can fail if only one class is predicted
    
    # majority-class baseline for comparison:
    baseline_pred = np.ones(len(y_val))
    print("\nmajority class baseline (always predicts error):")
    evaluate_model(y_val, baseline_pred)
    
    # Create results directory
    results_dir = BASE_PATH / "results" / "05bis_detailed"
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving all results to: {results_dir}")
    
    # Save baseline confusion matrix
    cm_baseline = confusion_matrix(y_val, baseline_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm_baseline, annot=True, fmt="d", cmap="Blues")
    plt.title("Baseline (Always Predicts Error)")
    plt.xlabel("predicted")
    plt.ylabel("true")
    plt.tight_layout()
    plt.savefig(results_dir / "00_baseline_confusion_matrix.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Saved baseline confusion matrix")
    
    # ========================================
    # logistic regression:
    # ========================================
    print("\n" + "="*80)
    print("logistic regression with gridsearch")
    print("="*80)
    
    log_params = {
        "C": [0.01, 0.1, 1, 5, 10],
        "penalty": ["l2"],
        "solver": ["lbfgs"],
    }
    
    log_grid = GridSearchCV(
        LogisticRegression(max_iter=10000, random_state=42),
        log_params,
        cv=3,
        scoring="f1",
        n_jobs=-1
    )
    
    log_grid.fit(X_train, y_train)
    
    best_log = log_grid.best_estimator_
    pred_log = best_log.predict(X_val)
    
    print("best params:", log_grid.best_params_)
    print("\nlogistic regression results:")
    evaluate_model(y_val, pred_log)
    
    # Save logistic confusion matrix
    cm_log = confusion_matrix(y_val, pred_log)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm_log, annot=True, fmt="d", cmap="Greens")
    plt.title("Logistic Regression")
    plt.xlabel("predicted")
    plt.ylabel("true")
    plt.tight_layout()
    plt.savefig(results_dir / "01_logistic_confusion_matrix.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Saved logistic confusion matrix")
    
    print("\n# embeddings massively improve the model")
    print("# accuracy goes from 0.667 to ~0.74 and global F1 also increases")
    print("# most importantly, the model learns to detect the minority class: F1 for class 0 jumps from 0.01 to ~0.56")
    print("# embeddings give the model a clear and useful signal instead of weak basic features")
    
    # ========================================
    # Knn:
    # ========================================
    print("\n" + "="*80)
    print("knn with gridsearch")
    print("="*80)
    
    knn_params = {
        "n_neighbors": [3, 5, 7, 9, 11],
        "weights": ["uniform", "distance"],
        "metric": ["euclidean", "manhattan"]
    }
    
    knn_grid = GridSearchCV(
        KNeighborsClassifier(),
        knn_params,
        cv=3,
        scoring="f1",
        n_jobs=-1
    )
    
    knn_grid.fit(X_train, y_train)
    
    best_knn = knn_grid.best_estimator_
    pred_knn = best_knn.predict(X_val)
    
    print("best params:", knn_grid.best_params_)
    print("\nknn results:")
    evaluate_model(y_val, pred_knn)
    
    # Save KNN confusion matrix
    cm_knn = confusion_matrix(y_val, pred_knn)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm_knn, annot=True, fmt="d", cmap="Purples")
    plt.title("KNN")
    plt.xlabel("predicted")
    plt.ylabel("true")
    plt.tight_layout()
    plt.savefig(results_dir / "02_knn_confusion_matrix.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Saved KNN confusion matrix")
    
    print("\n# embeddings improve Knn nd accuracy increases for every k:")
    print("# k=3 goes from 0.58 to ~0.68")
    print("# k=5 from 0.59 to ~0.70")
    print("# k=7 from 0.603 to ~0.71")
    print("# F1-score for the minority class also improves from ~ 0.30 to ~ 0.48 across all k")
    print("# confusion matrices show fewer false positives and more balanced behaviour instead of always predicting class 1")
    print("# distances become more meaningful thanks to the 512 dimensional embeddings")
    
    # ========================================
    # randomforest:
    # ========================================
    print("\n" + "="*80)
    print("random forest with gridsearch")
    print("="*80)
    
    rf_params = {
        "n_estimators": [200, 400, 600],
        "max_depth": [None, 10, 20, 30],
        "max_features": ["sqrt", "log2"]
    }
    
    rf_grid = GridSearchCV(
        RandomForestClassifier(random_state=42),
        rf_params,
        cv=3,
        scoring="f1",
        n_jobs=-1
    )
    
    rf_grid.fit(X_train, y_train)
    
    best_rf = rf_grid.best_estimator_
    pred_rf = best_rf.predict(X_val)
    
    print("best params:", rf_grid.best_params_)
    print("\nrandom forest results:")
    evaluate_model(y_val, pred_rf)
    
    # Save RF confusion matrix
    cm_rf = confusion_matrix(y_val, pred_rf)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm_rf, annot=True, fmt="d", cmap="Oranges")
    plt.title("Random Forest")
    plt.xlabel("predicted")
    plt.ylabel("true")
    plt.tight_layout()
    plt.savefig(results_dir / "03_rf_confusion_matrix.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Saved RF confusion matrix")
    
    print("\n# embeddings makes random forest go from 0.6748 to ~0.74")
    print("# the trees benefit from the high dimensional embedding space")
    print("# minority-class (correct predictions = class 0) F1 jumps from 0.31 to ~0.53")
    print("# precision nd recall also improve for both classes")
    print("# confusion matrix shows more true 0 are recovered 152 to ~314")
    print("# overall: embeddings give the trees real structure to learn from")
    
    # ========================================
    # support vector machine -> SVM :
    # ========================================
    print("\n" + "="*80)
    print("svm linear with gridsearch")
    print("="*80)
    
    svm_linear_params = {
        "C": [0.1, 0.5, 1, 2, 5, 10]
    }
    
    svm_linear_grid = GridSearchCV(
        SVC(kernel="linear", random_state=42),
        svm_linear_params,
        cv=3,
        scoring="f1",
        n_jobs=-1
    )
    
    svm_linear_grid.fit(X_train, y_train)
    
    best_svm_linear = svm_linear_grid.best_estimator_
    pred_svm_linear = best_svm_linear.predict(X_val)
    
    print("best params:", svm_linear_grid.best_params_)
    print("\nsvm linear results:")
    evaluate_model(y_val, pred_svm_linear)
    
    # Save SVM Linear confusion matrix
    cm_svm_lin = confusion_matrix(y_val, pred_svm_linear)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm_svm_lin, annot=True, fmt="d", cmap="Reds")
    plt.title("SVM Linear")
    plt.xlabel("predicted")
    plt.ylabel("true")
    plt.tight_layout()
    plt.savefig(results_dir / "04_svm_linear_confusion_matrix.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Saved SVM Linear confusion matrix")
    
    print("\n" + "="*80)
    print("svm rbf with gridsearch")
    print("="*80)
    
    svm_rbf_params = {
        "C": [0.1, 1, 5, 10],
        "gamma": [0.001, 0.01, 0.1, 1]
    }
    
    svm_rbf_grid = GridSearchCV(
        SVC(kernel="rbf", random_state=42),
        svm_rbf_params,
        cv=3,
        scoring="f1",
        n_jobs=-1
    )
    
    svm_rbf_grid.fit(X_train, y_train)
    
    best_svm_rbf = svm_rbf_grid.best_estimator_
    pred_svm_rbf = best_svm_rbf.predict(X_val)
    
    print("best params:", svm_rbf_grid.best_params_)
    print("\nsvm rbf results:")
    evaluate_model(y_val, pred_svm_rbf)
    
    # Save SVM RBF confusion matrix
    cm_svm_rbf = confusion_matrix(y_val, pred_svm_rbf)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm_svm_rbf, annot=True, fmt="d", cmap="YlOrRd")
    plt.title("SVM RBF")
    plt.xlabel("predicted")
    plt.ylabel("true")
    plt.tight_layout()
    plt.savefig(results_dir / "05_svm_rbf_confusion_matrix.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Saved SVM RBF confusion matrix")
    
    print("\n# embeddings improve Svm rbf from 0.667 to ~0.73 accuracy and F1 from 0.06 to ~0.55")
    print("# linear SVM stops collapsing before it was only class 1 -> now ~0.76 acc and real F1 = ~0.58")
    print("# both models finally learn class 0 instead of ignoring it")
    
    # ========================================
    # XGBoost with gridsearch
    # ========================================
    print("\n" + "="*80)
    print("xgboost with gridsearch")
    print("="*80)
    
    xgb_params = {
        "learning_rate": [0.01, 0.05, 0.1],
        "max_depth": [4, 6, 8],
        "n_estimators": [200, 400],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0]
    }
    
    xgb_grid = GridSearchCV(
        xgb.XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            random_state=42
        ),
        xgb_params,
        cv=3,
        scoring="f1",
        n_jobs=-1
    )
    
    xgb_grid.fit(X_train, y_train)
    best_xgb = xgb_grid.best_estimator_
    pred_xgb_tuned = best_xgb.predict(X_val)
    
    print("best params:", xgb_grid.best_params_)
    print("\nxgboost tuned results:")
    evaluate_model(y_val, pred_xgb_tuned)
    
    # Save XGBoost confusion matrix
    cm_xgb = confusion_matrix(y_val, pred_xgb_tuned)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm_xgb, annot=True, fmt="d", cmap="viridis")
    plt.title("XGBoost Tuned")
    plt.xlabel("predicted")
    plt.ylabel("true")
    plt.tight_layout()
    plt.savefig(results_dir / "06_xgboost_confusion_matrix.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Saved XGBoost confusion matrix")
    
    # ========================================
    # super ensemble: 6 models tuned with stacking
    # ========================================
    print("\n" + "="*80)
    print("super ensemble: 6 models tuned with stacking")
    print("="*80 + "\n")
    
    print("defining base models:")
    
    # using best params from gridsearch
    base_estimators = [
        ('logistic', best_log),
        ('knn', best_knn),
        ('rf', best_rf),
        ('svm_linear', SVC(kernel="linear", C=best_svm_linear.C, probability=True, random_state=42)),
        ('svm_rbf', SVC(kernel="rbf", C=best_svm_rbf.C, gamma=best_svm_rbf.gamma, probability=True, random_state=42)),
        ('xgboost', best_xgb)
    ]
    
    meta_model = LogisticRegression(max_iter=1000, random_state=42)
    
    print("\ncreating stacking ensemble")
    stacking_clf = StackingClassifier(
        estimators=base_estimators,
        final_estimator=meta_model,
        cv=5,
        stack_method='predict_proba',
        n_jobs=-1
    )
    
    print("training stacking ensemble")
    stacking_clf.fit(X_train, y_train)
    pred_stacking = stacking_clf.predict(X_val)
    
    print("\nstacking ensemble, 6 models")
    evaluate_model(y_val, pred_stacking)
    
    # Save Stacking confusion matrix
    cm_stack = confusion_matrix(y_val, pred_stacking)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm_stack, annot=True, fmt="d", cmap="coolwarm")
    plt.title("Stacking Ensemble (6 Models)")
    plt.xlabel("predicted")
    plt.ylabel("true")
    plt.tight_layout()
    plt.savefig(results_dir / "07_stacking_confusion_matrix.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Saved Stacking confusion matrix")
    
    # Create comprehensive results table
    results_table = pd.DataFrame({
        'Model': ['Baseline', 'Logistic', 'KNN', 'Random Forest', 'SVM Linear', 'SVM RBF', 'XGBoost', 'Stacking'],
        'Accuracy': [
            accuracy_score(y_val, baseline_pred),
            accuracy_score(y_val, pred_log),
            accuracy_score(y_val, pred_knn),
            accuracy_score(y_val, pred_rf),
            accuracy_score(y_val, pred_svm_linear),
            accuracy_score(y_val, pred_svm_rbf),
            accuracy_score(y_val, pred_xgb_tuned),
            accuracy_score(y_val, pred_stacking)
        ],
        'F1': [
            f1_score(y_val, baseline_pred),
            f1_score(y_val, pred_log),
            f1_score(y_val, pred_knn),
            f1_score(y_val, pred_rf),
            f1_score(y_val, pred_svm_linear),
            f1_score(y_val, pred_svm_rbf),
            f1_score(y_val, pred_xgb_tuned),
            f1_score(y_val, pred_stacking)
        ],
        'Precision': [
            precision_score(y_val, baseline_pred),
            precision_score(y_val, pred_log),
            precision_score(y_val, pred_knn),
            precision_score(y_val, pred_rf),
            precision_score(y_val, pred_svm_linear),
            precision_score(y_val, pred_svm_rbf),
            precision_score(y_val, pred_xgb_tuned),
            precision_score(y_val, pred_stacking)
        ],
        'Recall': [
            recall_score(y_val, baseline_pred),
            recall_score(y_val, pred_log),
            recall_score(y_val, pred_knn),
            recall_score(y_val, pred_rf),
            recall_score(y_val, pred_svm_linear),
            recall_score(y_val, pred_svm_rbf),
            recall_score(y_val, pred_xgb_tuned),
            recall_score(y_val, pred_stacking)
        ],
        'ROC_AUC': [
            0.5,  # baseline always predicts 1
            roc_auc_score(y_val, pred_log),
            roc_auc_score(y_val, pred_knn),
            roc_auc_score(y_val, pred_rf),
            roc_auc_score(y_val, pred_svm_linear),
            roc_auc_score(y_val, pred_svm_rbf),
            roc_auc_score(y_val, pred_xgb_tuned),
            roc_auc_score(y_val, pred_stacking)
        ]
    })
    
    # Save results table
    results_table.to_csv(results_dir / "model_comparison_table.csv", index=False)
    print("\n" + "="*80)
    print("FINAL RESULTS TABLE")
    print("="*80)
    print(results_table.to_string(index=False))
    print("\n✓ Saved results table to CSV")
    
    # Create comparison bar chart
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Accuracy
    axes[0,0].barh(results_table['Model'], results_table['Accuracy'], color='steelblue')
    axes[0,0].set_xlabel('Accuracy')
    axes[0,0].set_title('Model Accuracy Comparison')
    axes[0,0].grid(axis='x', alpha=0.3)
    
    # F1 Score
    axes[0,1].barh(results_table['Model'], results_table['F1'], color='forestgreen')
    axes[0,1].set_xlabel('F1 Score')
    axes[0,1].set_title('Model F1 Score Comparison')
    axes[0,1].grid(axis='x', alpha=0.3)
    
    # Precision
    axes[1,0].barh(results_table['Model'], results_table['Precision'], color='darkorange')
    axes[1,0].set_xlabel('Precision')
    axes[1,0].set_title('Model Precision Comparison')
    axes[1,0].grid(axis='x', alpha=0.3)
    
    # Recall
    axes[1,1].barh(results_table['Model'], results_table['Recall'], color='crimson')
    axes[1,1].set_xlabel('Recall')
    axes[1,1].set_title('Model Recall Comparison')
    axes[1,1].grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(results_dir / "08_model_comparison_chart.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Saved comparison chart")
    
    # ========================================
    # results embedding :
    # ========================================
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    print("\n# conclusion:")
    print("# in notebook 05, all models were trained only on the raw features scores, brightness, contrast, saturation")
    print("# and they all stayed close to the baseline: 0.667")
    print("# with the 512 dimensional embeddings, every model learns real structure and performance increases across the board")
    
    # save best models
    main_results_dir = BASE_PATH / "results"
    main_results_dir.mkdir(exist_ok=True)
    
    import pickle
    with open(main_results_dir / "scaler_with_embeddings.pkl", "wb") as f:
        pickle.dump(scaler, f)
    
    with open(main_results_dir / "xgboost_with_embeddings.pkl", "wb") as f:
        pickle.dump(best_xgb, f)
    
    with open(main_results_dir / "stacking_with_embeddings.pkl", "wb") as f:
        pickle.dump(stacking_clf, f)
    
    # Save all predictions for later analysis
    all_predictions = pd.DataFrame({
        'y_true': y_val,
        'baseline': baseline_pred.astype(int),
        'logistic': pred_log,
        'knn': pred_knn,
        'rf': pred_rf,
        'svm_linear': pred_svm_linear,
        'svm_rbf': pred_svm_rbf,
        'xgboost': pred_xgb_tuned,
        'stacking': pred_stacking
    })
    all_predictions.to_csv(results_dir / "all_predictions.csv", index=False)
    
    print("\n" + "="*80)
    print("ALL RESULTS SAVED!")
    print("="*80)
    print(f"\n✓ Models saved to: {main_results_dir}")
    print(f"✓ Detailed results saved to: {results_dir}")
    print(f"\nGenerated files:")
    print(f"  - 8 confusion matrices (PNG)")
    print(f"  - Model comparison table (CSV)")
    print(f"  - Model comparison chart (PNG)")
    print(f"  - All predictions (CSV)")
    print(f"  - Best models (PKL)")
    print("\nYou can now view all the confusion matrices and charts!")


if __name__ == "__main__":
    train_ml_with_embeddings()
