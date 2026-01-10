# AI Tool Usage Disclosure

This document describes how AI tools were used during the development of the FairFace Error Prediction project, in compliance with academic integrity guidelines.

---

## Tools Overview

I utilized a suite of Large Language Models as technical assistants. They were used interchangeably based on their strengths:

### 1. Claude (Anthropic) & ChatGPT (OpenAI)
**Purpose:** Code troubleshooting, syntax suggestions, and library documentation.
**How it was used:**
- Debugging specific DeepFace/TensorFlow integration errors.
- Suggesting efficient Pandas syntax for data manipulation.
- Explaining complex function parameters (e.g., XGBoost hyperparameters).

### 2. Gemini (Google)
**Purpose:** Reporting, translation, and formatting.
**How it was used:**
- Refining the professional tone of the report and README.
- Orthographic verification and summarizing key findings.

---

## Code Attribution & Development Process

To ensure academic integrity, I break down the specific contribution of AI versus my own work for each core component:

### 1. Data Preparation (`01_data_preparation.py`)
- **AI assisted with:** Suggesting efficient Pandas methods to avoid slow loops.
- **My contribution:** **Implemented** the data balancing algorithm and wrote the logic to split training/validation sets ensuring demographic parity.

### 2. Embeddings Extraction (`02BIS_extract_embeddings.py`)
- **AI assisted with:** Explaining the output format of Facenet512 to help me handle the vector dimensions.
- **My contribution:** **Wrote the script** to iterate through images, extract embeddings, and serialize the data into Parquet format for performance.

### 3. ML Models (`05BIS_predicting_with_embeddings.py`)
- **AI assisted with:** Debugging parameter names in the XGBoost grid search definition.
- **My contribution:** **Built the training pipeline**, defined the cross-validation strategy, and interpreted the F1-score trade-offs vs Accuracy.

### 4. Results & Visualization (`06_results.py`)
- **AI assisted with:** Suggesting aesthetic improvements for Matplotlib plots (color palettes as viridis).
- **My contribution:** **Coded the visualization logic** to compare "Before vs After" fairness metrics and generate the ROC curves.

---

## Learnings Moments

Through AI assistance and the development of this project, I deepened my understanding of:

1.  **Embeddings vs. Pixel Data:** I learned why 512-dimensional vectors capture semantic features (like bone structure) better than raw pixel brightness/contrast.
2.  **The "Accuracy Paradox":** I understood why high accuracy can be misleading in imbalanced datasets and why F1-score is crucial for error detection.
3.  **Taxonomy Bias:** I learned how a model's rigid classification labels (e.g., missing "Southeast Asian") create structural errors that no amount of retraining can fix without changing the labels.
4.  **Parquet Optimization:** I learned how to use `.parquet` files instead of `.csv` to handle large vector arrays efficiently.

---

## Code Understanding Confirmation

I confirm that I understand all code in this project, including:

- **Preprocessing:** How the dataset was balanced to 500 images per group to reduce sampling bias.
- **DeepFace pipeline:** How the `DeepFace.represent()` function operates and what the returned embedding vectors encode.
- **XGBoost logic:** Why a tree-based model (XGBoost) was preferred over linear models for high-dimensional embedding data.
- **Fairness metrics:** How the Demographic Parity reduction was computed and why it is relevant in this context.

---

## Declaration

I, Leo Bideau, author of this project, confirm that:
- I understand every line of code in this repository.
- AI tools were used as debugging aids and documentation assistants, not as ghostwriters.
- All analytical decisions, hypotheses, and interpretations are my own.
- I can explain any part of this project during the presentation.

*Date: December 2025*
