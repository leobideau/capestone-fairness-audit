# here the goal is to evaluate the results from notebook 2 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


def evaluate_baseline():
    
    BASE_PATH = Path(__file__).parent.parent
    
    sns.set(style="whitegrid")
    
    train_pred = pd.read_parquet(BASE_PATH / "results/baseline/pred_train.parquet")
    val_pred   = pd.read_parquet(BASE_PATH / "results/baseline/pred_val.parquet")
    
    print("Train predictions:", len(train_pred))
    print("Val predictions:",   len(val_pred))
    
    print(train_pred.head())
    # loading the deepface prediction files generated in notebook 2 and cheking the number of rows to confirm everything is fine
    
    train_gt = pd.read_csv(BASE_PATH / "data/processed/balanced_train.csv")
    val_gt   = pd.read_csv(BASE_PATH / "data/processed/balanced_val.csv")
    
    train = train_pred.merge(train_gt, on="file", how="left")
    val   = val_pred.merge(val_gt,   on="file", how="left")
    
    train = train.rename(columns={"gender": "gender_true", "race": "race_true"})
    val   = val.rename(columns={"gender": "gender_true", "race": "race_true"})
    
    print(len(train), len(val))
    print(train.head())
    # merging predictions with their true labels so I can measure accuracy nd bias
    
    gender_map = {
        "Man":   "Male",
        "Woman": "Female",
    }
    
    race_map = {
        "black":           "Black",
        "white":           "White",
        "asian":           "East Asian",     
        "indian":          "Indian",
        "latino hispanic": "Latino_Hispanic",
        "middle eastern":  "Middle Eastern",
        "others":          "Other",
    }
    
    train["pred_gender"] = train["pred_gender"].replace(gender_map)
    val["pred_gender"]   = val["pred_gender"].replace(gender_map)
    
    train["pred_race"] = train["pred_race"].replace(race_map)
    val["pred_race"]   = val["pred_race"].replace(race_map)
    
    print("unique predicted genders:", train["pred_gender"].unique())
    print("unique predicted races:",   train["pred_race"].unique())
    # map deepface label names to the label format used in my dataset so i can compare matching categories instead of mismatched strings
    
    print(train.columns)
    
    train = train.loc[:, ~train.columns.duplicated()]
    val   = val.loc[:, ~val.columns.duplicated()]
    
    print(train.columns)
    # remove duplicate columns accidentally created during the merge
    
    def add_error_cols(df):
        df["gender_ok"] = (df["pred_gender"] == df["gender_true"]).astype(int)
        df["race_ok"]   = (df["pred_race"] == df["race_true"]).astype(int)
        return df
    
    train = add_error_cols(train)
    val   = add_error_cols(val)
    
    print(train.head())
    # creating binary columns: 1 = correct, 0 = wrong -> these make accuracy calculations easier
    
    print("Train accuracy (gender):", train["gender_ok"].mean())
    print("Train accuracy (race):",   train["race_ok"].mean())
    
    print("\nVal accuracy (gender):", val["gender_ok"].mean())
    print("Val accuracy (race):",     val["race_ok"].mean())
    #check to see the global accuracy 
    
    race_acc = train.groupby("race_true")["race_ok"].mean().sort_values(ascending=False)
    gender_acc = train.groupby("gender_true")["gender_ok"].mean().sort_values(ascending=False)
    
    print("Race accuracy (train):")
    print(race_acc)
    
    print("\nGender accuracy (train):")
    print(gender_acc)
    # accuracy grouped by race
    
    print(train["pred_race"].unique())
    
    # deepface does not include 'Southeast Asian' as a race output, while fairface uses it as a true label
    # this causes deepface to systematically misclassify all Southeast Asian samples ->  0 in accuracy for this group
    
    # ============================================
    # BASELINE METRICS TABLE
    # ============================================
    print("\n" + "="*60)
    print("BASELINE PERFORMANCE METRICS (DeepFace - Race Classification)")
    print("="*60)
    
    # Create binary target: 1 = error, 0 = correct
    y_true_race = (val["race_true"] != val["pred_race"]).astype(int)
    
    # For ROC-AUC, we need probability scores
    if "pred_race_score" in val.columns:
        y_score_race = 1 - val["pred_race_score"]
        try:
            roc_auc_race = roc_auc_score(y_true_race, y_score_race)
        except:
            roc_auc_race = np.nan
    else:
        roc_auc_race = np.nan
    
    # Calculate metrics
    accuracy_race = accuracy_score(val["race_true"], val["pred_race"])
    
    # Calculate precision/recall for error prediction
    threshold = 0.5
    if "pred_race_score" in val.columns:
        y_pred_error = (val["pred_race_score"] < threshold).astype(int)
        precision_race = precision_score(y_true_race, y_pred_error, zero_division=0)
        recall_race = recall_score(y_true_race, y_pred_error, zero_division=0)
        f1_race = f1_score(y_true_race, y_pred_error, zero_division=0)
    else:
        precision_race = 0.0
        recall_race = 0.0
        f1_race = 0.0
    
    baseline_metrics = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'],
        'Score': [
            f"{accuracy_race:.4f}",
            f"{precision_race:.4f}",
            f"{recall_race:.4f}",
            f"{f1_race:.4f}",
            f"{roc_auc_race:.4f}" if not np.isnan(roc_auc_race) else "N/A"
        ]
    })
    
    print("\n" + baseline_metrics.to_string(index=False))
    print("\n" + "="*60)
    
    # Save to results folder
    results_dir = BASE_PATH / "results" / "baseline"
    results_dir.mkdir(parents=True, exist_ok=True)
    baseline_metrics.to_csv(results_dir / "baseline_metrics.csv", index=False)
    print(f"\n✓ Saved baseline metrics CSV: {results_dir / 'baseline_metrics.csv'}")
    
    # Create and save table image - EXACTLY like other plots
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.axis('off')
    
    table = ax.table(
        cellText=baseline_metrics.values,
        colLabels=['Metric', 'Score'],
        cellLoc='center',
        loc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    plt.title('DeepFace Baseline Performance Metrics', fontsize=12, pad=10)
    plt.tight_layout()
    plt.savefig(results_dir / 'baseline_metrics_table.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved baseline metrics image: {results_dir / 'baseline_metrics_table.png'}\n")
    plt.show()
    
    # ============================================
    # END BASELINE METRICS TABLE
    # ============================================
    
    cm_gender = confusion_matrix(val["gender_true"], val["pred_gender"])
    plt.figure(figsize=(5,4))
    sns.heatmap(cm_gender, annot=True, fmt="d", cmap="Blues",
                xticklabels=val["gender_true"].unique(),
                yticklabels=val["gender_true"].unique())
    plt.title("Confusion Matrix – Gender (Val)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()
    
    cm_race = confusion_matrix(val["race_true"], val["pred_race"])
    plt.figure(figsize=(7,6))
    sns.heatmap(cm_race, annot=True, fmt="d", cmap="Oranges",
                xticklabels=sorted(val["race_true"].unique()),
                yticklabels=sorted(val["race_true"].unique()))
    plt.title("Confusion Matrix – Race (Val)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()
    
    # confusion matrix - gender
    # it shows a strong biais: deepface predicts men far more easily than women
    # male faces are correctly classified most of the time but almost 1/2 of female faces are misclassified 
    
    # confusion matrix - race
    # deepface shows strong racial bias across groups
    # it performs well for east asian, white and black faces
    # it performs poorly for indian, middle eastern and latino faces
    # southeast asian has 0 percent accuracy because deepface does not include this group
    # this makes the model systematically misclassify southeast asian as east asian
    
    race_acc_val = val.groupby("race_true")["race_ok"].mean().sort_values()
    
    plt.figure(figsize=(8,5))
    sns.barplot(x=race_acc_val.index, y=race_acc_val.values, palette="viridis")
    plt.title("Validation Accuracy by Race")
    plt.ylabel("Accuracy")
    plt.xticks(rotation=45)
    plt.ylim(0,1)
    plt.show()
    
    gender_acc_val = val.groupby("gender_true")["gender_ok"].mean().sort_values()
    
    plt.figure(figsize=(5,4))
    sns.barplot(x=gender_acc_val.index, y=gender_acc_val.values, palette="viridis")
    plt.title("Validation Accuracy by Gender")
    plt.ylabel("Accuracy")
    plt.ylim(0,1)
    plt.show()
    
    # validation accuracy by race:
    # deepface performs best on east asian faces (~0.90), followed by black (~0.75) and white (~0.70)
    # all brown-skinned groups (indian, middle eastern, latino) fall below 0.35
    # southeast asian has 0 accuracy because the class does not exist in deepface
    
    # validation accuracy by gender:
    # gender accuracy is highly imbalanced: 
    # the model predicts mens correctly almost 90% of the time, while accuracy for womens is only around 58%
    # deepface systematically overpredicts 'Male'
    
    error_rate_race = 1 - race_acc_val
    
    plt.figure(figsize=(10,5))
    sns.barplot(x=error_rate_race.index, y=error_rate_race.values, palette="magma")
    plt.title("Validation Error Rate by Race")
    plt.ylabel("Error Rate")
    plt.ylim(0,1)
    plt.xticks(rotation=45)
    plt.show()
    
    # validation error rate by race:
    # error rates mirror the accuracy plot: east asian and black and white  have the lowest errors, while brown-skinned groups show very high one
    # southeast asian reaches a 100 percent error rate because the class does not exist in deepface and is always misclassified
    
    plt.figure(figsize=(10,5))
    sns.histplot(train["pred_race_score"], kde=True, bins=40)
    plt.title("Distribution of Race Prediction Confidence Scores")
    plt.xlabel("Confidence")
    plt.ylabel("Count")
    plt.show()
    
    # distribution of race prediction confidence:
    # the model shows a bimodal pattern with many low confidence guesses between 20 and 40 % and a huge spike at 100 %
    # deepface is often extremely overconfident even when it is wrong especially on minority group


if __name__ == "__main__":
    evaluate_baseline()
