# Supervised Machine Learning: Linear Regression on Diamonds Dataset

This project demonstrates the use of **Supervised Machine Learning** to predict diamond prices based on their features using **Linear Regression**. The dataset used is the `diamonds` dataset from the **Seaborn** library.

## Project Overview

The goal of this project is to build a linear regression model that predicts the price of diamonds based on various features such as carat, depth, table, and dimensions (`x`, `y`, `z`). The project also includes data preprocessing, feature engineering, model training, and evaluation.

## Dataset

The `diamonds` dataset is a built-in dataset in the Seaborn library. It contains the following features:

- **carat**: Weight of the diamond.
- **cut**: Quality of the cut (e.g., Fair, Good, Very Good, Premium, Ideal).
- **color**: Diamond color, from J (worst) to D (best).
- **clarity**: A measurement of how clear the diamond is (e.g., I1, SI1, VS1, etc.).
- **depth**: Total depth percentage = `z / mean(x, y) = 2 * z / (x + y)`.
- **table**: Width of the top of the diamond relative to the widest point.
- **x**: Length in mm.
- **y**: Width in mm.
- **z**: Depth in mm.
- **price**: Price in US dollars (target variable).

## Steps in the Project

1. **Data Loading**:
   - The `diamonds` dataset is loaded using `sns.load_dataset('diamonds')`.

2. **Data Exploration**:
   - The dataset is explored using methods like `.head()`, `.info()`, and `.describe()` to understand its structure and summary statistics.

3. **Feature Selection**:
   - The following features were selected for the model:
     - Numerical features: `carat`, `depth`, `table`, `x`, `y`, `z`.
     - Categorical feature: `clarity` (converted to dummy variables using `pd.get_dummies`).

4. **Data Splitting**:
   - The dataset is split into training and testing sets using `train_test_split` with an 80-20 split.

5. **Model Training**:
   - A **Linear Regression** model is trained using the `LinearRegression` class from `scikit-learn`.

6. **Model Evaluation**:
   - The model's performance is evaluated using:
     - **Mean Squared Error (MSE)**: Measures the average squared difference between actual and predicted prices.
     - **R-squared (R²)**: Indicates how well the model explains the variance in the target variable.

7. **Visualization**:
   - A scatter plot is created to compare actual vs. predicted prices, along with a line of equality for reference.

## Results

- **Mean Squared Error (MSE)**: The average squared error between the predicted and actual prices.
- **R-squared (R²)**: Indicates the proportion of variance in the target variable explained by the model.

## Dependencies

The following Python libraries are required to run the project:

- `pandas`
- `numpy`
- `seaborn`
- `matplotlib`
- `scikit-learn`

Install them using the following command:

```bash
! pip install -q pandas matplotlib seaborn scikit-learn numpy
```
