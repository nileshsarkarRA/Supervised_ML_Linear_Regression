## Logistic Regression Notebook

This repository demonstrates the implementation of Logistic Regression for classification tasks using three datasets: a **synthetic dataset**, the **IRIS dataset**, and a **spam detection dataset**. Each notebook provides a comprehensive walkthrough of the steps involved in data preparation, model training, evaluation, and visualization.

---

### **1. Logistic Regression on Synthetic Dataset**
The `Logistic_Regression_FLMN.ipynb` notebook demonstrates the implementation of Logistic Regression using a **synthetic dataset**. This dataset was generated using the `make_classification` function from `scikit-learn`, which allows for the creation of a controlled dataset with specific characteristics for binary classification problems. The notebook provides a comprehensive walkthrough of the following steps:

#### **Steps:**
1. **Data Generation**:
   - A synthetic dataset is created with:
     - **1000 samples**
     - **2 informative features**
     - **0 redundant features**
     - A **random state** for reproducibility.
   - The dataset is split into training and testing sets using an 80-20 split.

2. **Data Preprocessing**:
   - The features and labels are prepared for training.
   - The dataset is visualized to understand the distribution of the two classes.

3. **Model Training**:
   - A Logistic Regression model is instantiated and trained using the `LogisticRegression` class from `scikit-learn`.
   - The model is fit on the training data (`X_train`, `y_train`).

4. **Model Evaluation**:
   - The model's performance is evaluated using the following metrics:
     - **Confusion Matrix**
     - **Classification Report**
     - **Accuracy Score**

5. **Visualization**:
   - **Confusion Matrix Heatmap**
   - **Decision Boundary**

---

### **2. Logistic Regression on IRIS Dataset**
The `Logistic_Regression_Using_IRIS_Dataset.ipynb` notebook demonstrates the implementation of Logistic Regression using the **IRIS dataset**, a well-known dataset in machine learning. This dataset contains information about three species of iris flowers (`setosa`, `versicolor`, and `virginica`) and their respective features (`sepal length`, `sepal width`, `petal length`, `petal width`). The notebook provides a detailed walkthrough of the following steps:

#### **Steps:**
1. **Data Loading**
2. **Data Preprocessing**
3. **Data Splitting**
4. **Model Training**
5. **Model Evaluation**
6. **Visualization**

---

### **3. Logistic Regression for Spam Detection**
The `Project_Logistic_Regression` folder contains a project that applies Logistic Regression to a **spam detection** task. This project uses the `SMSSpamCollection.csv` dataset, which contains labeled SMS messages as either "spam" or "ham" (not spam). The notebook provides a detailed walkthrough of the following steps:

#### **Steps:**
1. **Data Loading**:
   - The dataset is loaded and inspected for missing values or inconsistencies.

2. **Data Preprocessing**:
   - Text data is cleaned and preprocessed using techniques such as:
     - Removing punctuation and stopwords.
     - Tokenization and stemming/lemmatization.
   - The text data is converted into numerical features using **TF-IDF Vectorization**.

3. **Model Training**:
   - A Logistic Regression model is trained on the preprocessed data.

4. **Model Evaluation**:
   - The model's performance is evaluated using:
     - **Confusion Matrix**
     - **Classification Report**
     - **Accuracy Score**

5. **Results**:
   - The project demonstrates how Logistic Regression can effectively classify SMS messages as spam or ham with high accuracy.

---

### **Conclusion**
This repository provides three practical examples of implementing Logistic Regression for classification tasks:
1. A synthetic dataset for binary classification.
2. The IRIS dataset for multi-class classification.
3. A spam detection dataset for binary classification.

Each project serves as an excellent resource for understanding the fundamentals of Logistic Regression and its application in real-world scenarios.