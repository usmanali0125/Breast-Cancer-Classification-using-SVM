# Breast Cancer Classification 🎗️

## Project Overview
This project applies machine learning techniques to classify breast cancer tumors as either Malignant (M) or Benign (B). By analyzing tumor features such as radius, texture, perimeter, and area, this model serves as a predictive tool to assist in medical diagnostics. 

This specific repository explores the **Support Vector Machine (SVM)** classification algorithm, focusing on maximizing the margin between classes in a high-dimensional feature space, and addresses class imbalance using advanced resampling techniques.

## Dataset
The data used in this project is the **Breast Cancer Wisconsin (Diagnostic) Dataset**. 
* **Target Variable:** `diagnosis` (M = Malignant, B = Benign)
* **Features:** 30 numerical features computed from a digitized image of a fine needle aspirate (FNA) of a breast mass (e.g., `radius_mean`, `texture_mean`, `smoothness_mean`).

## Tech Stack
* **Language:** Python
* **Libraries:** * `pandas` & `numpy` (Data manipulation)
  * `matplotlib` & `seaborn` (Data visualization)
  * `scikit-learn` (Machine learning modeling & evaluation)
  * `imbalanced-learn` (Handling imbalanced datasets)

## Methodology & Workflow

1. **Data Preprocessing & Scaling (CRITICAL):** * Dropped unnecessary columns (e.g., `id`).
   * Mapped categorical labels to binary values (`M` -> 1, `B` -> 0).
   * **Feature Scaling:** Because SVMs aim to maximize the distance (margin) between data points and the decision boundary, standardizing the features using `StandardScaler` was a mandatory step to prevent features with larger numeric ranges from dominating the algorithm.

2. **Handling Class Imbalance:**
   To prevent the model from becoming biased toward the majority class, a hybrid resampling approach was used:
   * **SMOTE (Synthetic Minority Over-sampling Technique):** Upsampled the minority class to 80% of the majority class size.
   * **RandomUnderSampler (RUS):** Downsampled the majority class to achieve a perfectly balanced 1:1 ratio.

3. **Modeling:**
   * Trained a **Support Vector Machine (SVM)**, adjusting regularization hyperparameters to find the optimal decision boundary (hyperplane) that separates malignant from benign tumors.

4. **Evaluation:**
   The model was evaluated on a dedicated test set using Accuracy, Precision, Recall, F1-Score, and Confusion Matrices.

## Results
*(Note: Replace the Xs with your final scores!)*

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| **Support Vector Machine (SVM)** | 97% | 98% | 99% | 97% |

**Key Takeaways:** * Feature scaling dramatically improved the SVM's ability to find an accurate decision boundary.
* Implementing the SMOTE + RUS hybrid technique successfully reduced false negatives, which is critical in medical diagnostics where missing a malignant tumor is highly dangerous.

## How to Run This Project
1. Clone the repository:
   ```bash
   git clone [https://github.com/usmanali0125/Breast-Cancer-Classification-using-SVM.git]

Install the required dependencies:
pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn


Open the Jupyter Notebook to explore the code and run the cells:
jupyter notebook "SVM.ipynb"
