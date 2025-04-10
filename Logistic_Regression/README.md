## Logistic Regression Notebook

This repository demonstrates the implementation of Logistic Regression for classification tasks using two datasets: a **synthetic dataset** and the **IRIS dataset**. Both notebooks provide a comprehensive walkthrough of the steps involved in data preparation, model training, evaluation, and visualization.

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
     - **Confusion Matrix**:
       - A confusion matrix is generated to evaluate the number of true positives, true negatives, false positives, and false negatives.
       - A heatmap is plotted for better visualization.
     - **Classification Report**:
       - Includes precision, recall, F1-score, and support for each class.
     - **Accuracy Score**:
       - The overall accuracy of the model is calculated.

5. **Visualization**:
   - **Confusion Matrix Heatmap**:
     - A heatmap is plotted to visualize the confusion matrix.
   - **Decision Boundary**:
     - The decision boundary of the Logistic Regression model is plotted using a mesh grid.
     - The plot shows how the model separates the two classes in the feature space.

6. **Key Concepts Explained**:
   - **Logistic Regression Overview**:
     - Logistic Regression is a supervised machine learning algorithm used for classification tasks. It predicts the probability of an instance belonging to a specific class (e.g., Class 0 or Class 1).
     - The algorithm uses the **sigmoid function** to map input features to a probability value between 0 and 1.
   - **Confusion Matrix**:
     - The confusion matrix is explained in detail, including the definitions of True Positive (TP), True Negative (TN), False Positive (FP), and False Negative (FN).
   - **Classification Metrics**:
     - **Accuracy**: Measures the overall correctness of the model.
     - **Precision**: Measures how many of the predicted positive cases are actually positive.
     - **Recall (Sensitivity)**: Measures how many of the actual positive cases are correctly predicted.
     - **F1-Score**: Harmonic mean of precision and recall.

7. **Example Use Case**:
   - The notebook demonstrates how Logistic Regression can be used for binary classification tasks, such as spam detection or predicting whether an email is spam (Class 1) or not (Class 0).

---

### **2. Logistic Regression on IRIS Dataset**
The `Logistic_Regression_Using_IRIS_Dataset.ipynb` notebook demonstrates the implementation of Logistic Regression using the **IRIS dataset**, a well-known dataset in machine learning. This dataset contains information about three species of iris flowers (`setosa`, `versicolor`, and `virginica`) and their respective features (`sepal length`, `sepal width`, `petal length`, `petal width`). The notebook provides a detailed walkthrough of the following steps:

#### **Steps:**
1. **Data Loading**:
   - The IRIS dataset is loaded using the `load_iris` function from `scikit-learn`.

2. **Data Preprocessing**:
   - The dataset is converted into a Pandas DataFrame for easier manipulation.
   - The target variable (`Species`) is added to the DataFrame, and its numerical values are mapped to species names (`setosa`, `versicolor`, `virginica`).
   - Intermediate results are saved into three Excel files:
     - `iris_1.xlsx`: Contains only the feature columns.
     - `iris_2.xlsx`: Includes the `Species` column with numerical values.
     - `iris_3.xlsx`: Includes the `Species` column with species names.

3. **Data Splitting**:
   - The dataset is split into training (30%) and testing (70%) sets using `train_test_split`.

4. **Model Training**:
   - A Logistic Regression model is instantiated and trained on the training data (`X_train`, `y_train`).

5. **Model Evaluation**:
   - The model's performance is evaluated using:
     - **Confusion Matrix**:
       - A confusion matrix is generated to evaluate the number of true positives, true negatives, false positives, and false negatives.
     - **Classification Report**:
       - Includes precision, recall, F1-score, and support for each class.
     - **Accuracy Score**:
       - The overall accuracy of the model is calculated.

6. **Visualization**:
   - **Confusion Matrix Heatmap**:
     - A heatmap is plotted to visualize the confusion matrix.

7. **Results**:
   - The notebook demonstrates how Logistic Regression can classify the three species of iris flowers with high accuracy.


### Explanation of Data Preprocessing in Logistic Regression on the IRIS Dataset

#### **1. Converting the Dataset into a Pandas DataFrame**
- The IRIS dataset is loaded using the `load_iris` function from `scikit-learn`. This dataset is initially in the form of a dictionary-like object.
- The feature data (sepal length, sepal width, petal length, petal width) is extracted and converted into a Pandas DataFrame. This makes it easier to manipulate and analyze the data using Pandas' powerful data handling capabilities.

#### **2. Adding the Target Variable (`Species`)**
- The target variable (`iris.target`) is added as a new column named `Species` in the DataFrame. This column contains numerical values (`0`, `1`, `2`) representing the three species of iris flowers:
  - `0` → `setosa`
  - `1` → `versicolor`
  - `2` → `virginica`

#### **3. Mapping Numerical Values to Species Names**
- The numerical values in the `Species` column are replaced with their corresponding species names using the `map()` function. This makes the data more interpretable and easier to understand.

#### **4. Saving Intermediate Results into Excel Files**
- The processed data is saved into three separate Excel files at different stages of preprocessing:
  - **`iris_1.xlsx`**:
    - Contains only the feature columns (`sepal length`, `sepal width`, `petal length`, `petal width`).
    - This represents the raw feature data without the target variable.
  - **`iris_2.xlsx`**:
    - Includes the `Species` column with numerical values (`0`, `1`, `2`).
    - This file represents the dataset after adding the target variable in its numerical form.
  - **`iris_3.xlsx`**:
    - Includes the `Species` column with species names (`setosa`, `versicolor`, `virginica`).
    - This file represents the final processed dataset, where the target variable is mapped to human-readable species names.

#### **Purpose of These Steps**
- These preprocessing steps ensure that the data is clean, well-structured, and ready for analysis or model training.
- Saving intermediate results into Excel files provides checkpoints for reviewing the data at different stages of preprocessing, which can be useful for debugging or sharing the data with others.

---

### **Conclusion**
This repository provides two practical examples of implementing Logistic Regression for classification tasks:
1. A synthetic dataset for binary classification.
2. The IRIS dataset for multi-class classification.

Both notebooks serve as excellent resources for understanding the fundamentals of Logistic Regression and its application in real-world scenarios.