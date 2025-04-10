# Supervised Machine Learning Projects

This directory contains projects demonstrating the application of **Supervised Machine Learning** techniques to solve real-world problems. The focus is on **Linear Regression** and **Logistic Regression**, showcasing the end-to-end process of data preprocessing, model training, evaluation, and visualization.

---

## Overview

Supervised Machine Learning involves training a model on labeled data to make predictions or classifications. This repository includes two projects:

1. **Linear Regression**: Predicting continuous outcomes.
2. **Logistic Regression**: Performing binary and multi-class classification.

---

## Projects

### 1. Linear Regression
- **Description**: Predicts the price of diamonds using the `diamonds` dataset from the **Seaborn** library.
- **Key Features**:
  - Data preprocessing, including handling categorical variables and feature scaling.
  - Model training using the `LinearRegression` class from `scikit-learn`.
  - Evaluation metrics: **Mean Squared Error (MSE)** and **R-squared (R²)**.
  - Visualization of actual vs. predicted prices.
- **Directory Structure**:
  ```
  Linear_Regression/
  ├── Linear_Regression_FLMN.ipynb  # Jupyter Notebook for Linear Regression
  └── README.md                     # Documentation for the project
  ```
- **Location**: [Linear_Regression](./Linear_Regression)

---

### 2. Logistic Regression
- **Description**: Demonstrates binary and multi-class classification using logistic regression.
- **Key Features**:
  - **Synthetic Dataset**:
    - Data generation using `make_classification` from `scikit-learn`.
    - Model evaluation using metrics like **accuracy**, **confusion matrix**, and **classification report**.
    - Visualization of decision boundaries.
  - **IRIS Dataset**:
    - Multi-class classification of iris flower species.
    - Model evaluation using metrics like **accuracy** and **classification report**.
  - **Spam Detection**:
    - Binary classification of SMS messages as spam or not spam.
    - Text preprocessing using **TF-IDF Vectorization**.
    - Model evaluation using metrics like **accuracy**, **precision**, and **recall**.
- **Directory Structure**:
  ```
  Logistic_Regression/
  ├── Logistic_Regression_Using_Synthetic_Dataset.ipynb  # Logistic Regression on synthetic data
  ├── Logistic_Regression_Using_IRIS_Dataset.ipynb       # Logistic Regression on IRIS dataset
  ├── Project_Logistic_Regression/
  │   ├── spam_detection_logic.ipynb                     # Spam detection project
  │   └── SMSSpamCollection.csv                          # Dataset for spam detection
  └── README.md                                          # Documentation for Logistic Regression
  ```
- **Location**: [Logistic_Regression](./Logistic_Regression)

---

## Dependencies

The following Python libraries are required to run the projects:

- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`

Install them using the following command:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```