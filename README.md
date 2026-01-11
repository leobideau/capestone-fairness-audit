# FairFace - Predicting DeepFace Errors with Machine Learning

This project analyzes racial and gender biases in the DeepFace facial recognition model and develops a machine learning system to predict when DeepFace will misclassify faces. This work addresses a critical fairness problem: pre-trained models like DeepFace exhibit systematic errors on underrepresented demographic groups.

## Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/leobideau/capestone-fairness-audit.git
cd fairface-project
```

### 2. Download the data folder

**Download the zip file from Google Drive:** https://drive.google.com/file/d/1KmzcO3jcE1zVn-1L4JvDD9jdlPH9rTyf/view?usp=sharing

Place the `data/` folder at the root of the project:

```
fairface-project/
├── data/          <-- unzip it and put it here
├── notebooks/
├── src/
└── ...
```

### 3. Setup environment

```bash
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 4. Run the pipeline

```bash
python main.py
```

**Note**: The complete pipeline execution requires approximately 1.5 to 2 hours, primarily due to DeepFace predictions, XGBoost hyperparameter tuning, and model stacking. To significantly reduce runtime, I have included pre-computed `.parquet` files in the data directory. Without these intermediate files, execution would take approximately 5–6 hours.

---

## Project Overview

**Goal**: Build a system that predicts when DeepFace will make errors, particularly on underrepresented demographic groups.

**Key Results**:

- **Baseline Failure**: DeepFace accuracy falls to 46.2% (with 0% accuracy on Southeast Asian faces due to taxonomy defects).
- **High-Performance Detection**: The XGBoost model achieves an F1-score of 0.83 in flagging these errors.
- **Systemic Improvement**: Fairness disparity is reduced by 74%, significantly closing the performance gap between demographic groups.

---

## Dependencies

- Python 3.11
- All required libraries with versions are listed in `requirements.txt`
- Key dependencies: `pandas`, `scikit-learn`, `deepface`, `xgboost`, `tensorflow`, `opencv-python`

---

## Detailed Pipeline Description

### 1. Data Preparation (01_data_preparation.py)

Balances the FairFace dataset by race and gender to ensure equal representation. Creates 500 train and 150 validation samples per demographic group (7 races x 2 genders = 14 groups). Copies balanced images to the processed directory, resulting in a total of 7,000 training and 2,100 validation images.

### 2. Baseline DeepFace Predictions (02_baseline_predictions.py)

Runs DeepFace model on all balanced images using RetinaFace detector. Extracts race and gender predictions with confidence scores. Saves predictions to parquet files with checkpointing every 50 images to handle potential crashes.

### 3. Embeddings Extraction (02BIS_extract_embeddings.py)

Extracts 512-dimensional face embeddings using Facenet512 model. These dense representations capture facial characteristics beyond basic features and enable better error prediction.

### 4. Baseline Evaluation (03_baseline_evaluation.py)

Evaluates DeepFace performance by demographic group. Generates confusion matrices and accuracy plots. Reveals systematic bias: 46.19% overall accuracy, 0% on Southeast Asian (class not in DeepFace taxonomy), and a 90.7 point disparity between best and worst groups.

### 5. Feature Engineering (04_feature_engineering.py)

Extracts basic image features from face crops: brightness, contrast, and saturation using OpenCV. Merges features with DeepFace predictions and ground truth labels to create ML-ready datasets.

### 6. Merge Features + Embeddings (04BIS_merge_embeddings.py)

Combines basic features (brightness, contrast, saturation) with Facenet512 embeddings. Expands embedding vectors into 512 separate columns and concatenates with existing features to create final ML datasets (524 features total).

### 7. ML Error Prediction - Baseline (05_ML_error_prediction.py)

Trains four baseline ML models (Logistic Regression, KNN, Random Forest, SVM) using only basic features. All models perform poorly (58-67% accuracy), barely exceeding majority-class baseline, demonstrating that simple features are insufficient.

### 8. ML Error Prediction - With Embeddings (05BIS_predicting_with_embeddings.py)

Trains models using features + embeddings. XGBoost with hyperparameter tuning achieves an F1-score of 0.8344. This allows the final system to reach 67.33% accuracy across all groups and detects 69.33% of Southeast Asian misclassifications despite DeepFace's 0% accuracy on this group.

### 9. Results & Fairness Analysis (06_results.py)

Generates comparison visualizations showing before/after fairness metrics. Calculates disparity reduction from 90.7 to 23.3 points (-74.3%). Produces ROC curves and model performance tables across all demographic groups.

### 10. Case Analysis (07_cases_analysis.py)

Analyzes three failure modes: hard cases where both DeepFace and XGBoost fail, false positives where XGBoost incorrectly flags correct predictions, and structural bias examples showing Southeast Asian detection patterns.

---

## Folder Structure

```
fairface-project/
├── data/                        # Download from Google Drive
│   ├── raw/                     # FairFace original dataset
│   ├── processed/               # Balanced dataset
│   ├── embeddings/              # Facenet512 embeddings
│   ├── ml_ready/                # Features without embeddings
│   └── ml_final/                # Features + embeddings
├── notebooks/                   # Jupyter notebooks (for exploration)
├── src/                         # Python scripts
│   ├── 01_data_preparation.py
│   ├── 02_baseline_predictions.py
│   ├── 02BIS_extract_embeddings.py
│   ├── 03_baseline_evaluation.py
│   ├── 04_feature_engineering.py
│   ├── 04BIS_merge_embeddings.py
│   ├── 05_ML_error_prediction.py
│   ├── 05BIS_predicting_with_embeddings.py
│   ├── 06_results.py
│   └── 07_cases_analysis.py
├── results/                     # Output visualizations and models
│   ├── baseline/                # Baseline outputs and .parquets (pre-computed to save time)
│   │   ├── pred_train.parquet
│   │   └── pred_val.parquet
│   │   └── *.png                # Visualizations
│   └── features/                # basic features.parquets (pre-computed to save time)
│   │   ├── train_features.parquet
│   │   └── val_features.parquet
│   └── *.png 
│                   
├── main.py                      # Pipeline orchestrator
├── requirements.txt             # Python dependencies
├── AI_USAGE.md                  # AI tools documentation
├── README.md			 
├── PROPOSAL.md            	 
└── gitignore

```

---

## Graphs Produced

- **race_comparison_before_after.png**: DeepFace baseline vs XGBoost accuracy by race
- **gender_comparison_before_after.png**: Performance improvement by gender
- **grouped_comparison_before_after.png**: Combined race-gender group analysis
- **race_improvement.png**: Absolute accuracy gains per racial group
- **fairness_metrics_comparison.png**: Disparity reduction visualization
- **failure_cases_both_wrong.png**: Hard cases where both models fail
- **false_positives.png**: Cases where XGBoost incorrectly flags errors
- **southeast_asian_bias.png**: Structural bias examples with detection confidence levels
- **roc_curves_comparison.png**: ROC curves for all models

---

## Discussion

DeepFace fails completely on Southeast Asian faces (0% accuracy) because its taxonomy only includes "Asian" without distinguishing East Asian from Southeast Asian. This taxonomic limitation creates a structural bias that cannot be fixed without retraining DeepFace itself.

The XGBoost model with Facenet512 embeddings partially mitigates this by learning to predict when DeepFace will fail, achieving 69.33% detection accuracy on Southeast Asian errors. However, the system remains imperfect: 30% of Southeast Asian errors go undetected, and false positives occur on correctly classified faces. The 74.3% disparity reduction represents significant but incomplete progress toward fairness.

---

## Conclusion

This project demonstrates that machine learning error prediction can significantly reduce demographic bias in pre-trained facial recognition systems. The optimized XGBoost model with embeddings improves overall accuracy from 46.19% to 67.33% across balanced demographic groups and reduces group disparity by 74.3%. Crucially, it successfully detects errors on a demographic group (Southeast Asian) that was completely missing from the base model's taxonomy. These results confirm the viability of post-hoc fairness interventions when retraining large-scale models is not feasible.

---

## Author

**Name**: Leo Bideau  
**Student ID**: 21434774

Date: December 2025

