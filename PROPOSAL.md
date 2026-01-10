# Project Proposal: Predicting Facial Recognition Failures with ML

**Student:** Léo Bideau, ID: 21434774

## 1. Introduction & Motivation
Facial recognition systems are widely deployed in critical sectors but often exhibit significant performance disparities across demographic groups. While auditing these models to quantify bias is a necessary first step, simply measuring error rates is insufficient for building robust, fair systems.

This project pivots from a traditional fairness audit to an active **Machine Learning Engineering task**. Following the "Predict Bias Patterns" approach, the goal is to build a secondary "safety net" model capable of predicting *when* a primary facial recognition system (DeepFace) is likely to fail, thereby transforming an evaluation task into a supervised learning problem.

## 2. Research Question
Can we train a post-hoc machine learning classifier to accurately anticipate algorithmic failures in pre-trained models like DeepFace using observable features (such as image characteristics and confidence scores), without needing to retrain the underlying architecture?

## 3. Dataset Description
I will utilize the **FairFace** dataset.
* **Source:** A public dataset labeled with race, gender, and age, providing a balanced alternative to traditional computer vision datasets.
* **Preprocessing:** To ensure the error predictor learns genuine failure patterns rather than memorizing class frequencies, the dataset will be re-balanced to ensure statistical parity across racial groups and genders before analysis.

## 4. Methodology
The project will follow a three-phase pipeline:

### Phase 1: The Audit (Generating Ground Truth)
* Run a pre-trained **DeepFace** model on the balanced dataset.
* Compare predictions against ground truth labels to generate a binary target variable: `0` (Correct Prediction) vs `1` (Error). This created dataset will serve as the training ground for the error predictor.

### Phase 2: Feature Engineering
To train the predictive model, I will extract various input features as suggested in the project guidelines:
* **Model Confidence:** Utilizing the probability outputs from DeepFace to test if low confidence correlates with failure.
* **Image Characteristics:** Extracting numerical data representing the visual content (e.g., pixel statistics or metadata) to determine if specific conditions drive the algorithmic bias.

### Phase 3: Model Training (The "Build" Phase)
I will train supervised learning classifiers to predict the probability of error.
* **Models:** I plan to benchmark several standard algorithms (e.g., Logistic Regression, Random Forest) to find the best architecture for this binary classification task.
* **Evaluation:** Performance will be evaluated using metrics suitable for error detection (such as Precision, Recall, or ROC-AUC) to assess the model's reliability as a filter.

## 5. Expected Outcomes
* A trained machine learning model capable of flagging potential DeepFace errors before they are accepted.
* An analysis identifying which input features are most predictive of model failure.
* A demonstration of how this "wrapper" approach can reduce fairness disparities in deployment.

## 6. Tools & Tech Stack
* **Language:** Python.
* **Libraries:** `DeepFace`, `scikit-learn`, `pandas`, `numpy`.
* **Environment:** Development will follow standard software engineering practices (PEP8, reproducibility).

## 7. Dataset Source
**Repository:** https://github.com/joojs/fairface?tab=readme-ov-file