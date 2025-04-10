# Logistic Regression: Spam Detection

This project demonstrates the use of **Logistic Regression**, a supervised machine learning algorithm, to classify emails as either **spam** or **not spam**. The dataset used contains labeled email data with features extracted from the email content.

---

## Project Overview

The goal of this project is to build a **binary classification model** that predicts whether an email is spam (1) or not spam (0). The project involves data preprocessing, feature engineering, model training, evaluation, and visualization of results.

---

## Dataset

The dataset used for this project contains the following:

- **Features**: Numerical and categorical features extracted from email content, such as:
  - Frequency of specific words (e.g., "free", "win").
  - Frequency of special characters (e.g., "!", "$").
  - Length of the email.
  - Presence of certain keywords.
- **Target Variable**: A binary label:
  - `1` for spam emails.
  - `0` for non-spam emails.

---

## Steps in the Project

1. **Data Loading**:
   - The dataset is loaded using `pandas` for analysis and preprocessing.

2. **Data Exploration**:
   - The dataset is explored using `.head()`, `.info()`, and `.describe()` to understand its structure and summary statistics.

3. **Data Preprocessing**:
   - Missing values are handled.
   - Features are scaled using `StandardScaler` to normalize the data.
   - Categorical features are encoded using one-hot encoding.

4. **Feature Selection**:
   - Features that contribute most to the classification task are selected using techniques like correlation analysis or feature importance scores.

5. **Data Splitting**:
   - The dataset is split into training and testing sets using `train_test_split` with an 80-20 split.

6. **Model Training**:
   - A **Logistic Regression** model is trained using the `LogisticRegression` class from `scikit-learn`.

7. **Model Evaluation**:
   - The model's performance is evaluated using:
     - **Accuracy**: Percentage of correctly classified emails.
     - **Precision**: Proportion of true spam emails among all emails classified as spam.
     - **Recall**: Proportion of correctly identified spam emails among all actual spam emails.
     - **F1-Score**: Harmonic mean of precision and recall.

8. **Visualization**:
   - A confusion matrix is plotted to visualize the model's performance.
   - ROC curve and AUC score are used to evaluate the model's ability to distinguish between spam and non-spam emails.

---

## Results

- **Accuracy**: The model achieved an accuracy of `XX%` on the test set.
- **Precision**: `YY%`
- **Recall**: `ZZ%`
- **F1-Score**: `AA%`
- **AUC Score**: `BB`

---

## Dependencies

The following Python libraries are required to run the project:

- `pandas`
- `numpy`
- `scikit-learn`
- `matplotlib`
- `seaborn`

Install them using the following command:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn


