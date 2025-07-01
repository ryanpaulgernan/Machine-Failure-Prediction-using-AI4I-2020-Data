# ⚙️ Machine Failure Prediction Project

## Table of Contents
1.  [Project Overview](#1-project-overview)
2.  [Features and Target Variable](#2-features-and-target-variable)
3.  [Installation](#3-installation)
4.  [Data Preparation and Preprocessing](#4-data-preparation-and-preprocessing)
5.  [Exploratory Data Analysis (EDA)](#5-exploratory-data-analysis-eda)
6.  [Machine Learning Model Development](#6-machine-learning-model-development)
7.  [Model Evaluation Metrics](#7-model-evaluation-metrics)
8.  [How to Run the Streamlit Application](#8-how-to-run-the-streamlit-application)
9. [Future Enhancements](#9-future-enhancements)
10. [Contact](#10-contact)

## 1. Project Overview

This project focuses on building a machine learning solution to predict potential machine failures or breakdowns based on various sensor readings and operational parameters. By leveraging predictive maintenance principles, the goal is to enable proactive interventions, thereby reducing costly downtime, optimizing maintenance schedules, and enhancing overall operational efficiency and safety.

The core of the project involves:
* Collecting and cleaning sensor data.
* Performing extensive Exploratory Data Analysis (EDA) to understand data distributions and relationships.
* Training and evaluating multiple classification models suitable for imbalanced datasets.
* Deploying the best-performing model (or a strong baseline) as an interactive web application using Streamlit.

## 2. Features and Target Variable

The dataset comprises **Machine Health Indicators** and a binary target variable:

### Input Features (Machine Health Indicators):
* `Air Temperature [K]`: Ambient air temperature around the machine.
* `Process Temperature [K]`: Temperature of the machine's internal processes.
* `Rotational Speed [rpm]`: Rotations per minute of the machine's primary rotating component.
* `Torque [Nm]`: The rotational force exerted by the machine.
* `Tool Wear [min]`: Cumulative wear on the machine's cutting tools (if applicable and used in the model).

### Target Variable:
* `Machine failure` (Binary: `0` or `1`):
    * `0`: No failure
    * `1`: Machine failure/breakdown

This is a **binary classification problem**, where the challenge lies in effectively predicting the relatively rare `1` (failure) class.

## 3. Installation

To set up the project locally, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/machine-failure-prediction.git](https://github.com/your-username/machine-failure-prediction.git)
    cd machine-failure-prediction
    ```
    *(Replace `https://github.com/your-username/machine-failure-prediction.git` with your actual repository URL)*

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv .venv
    # On Windows:
    .\.venv\Scripts\activate
    # On macOS/Linux:
    source ./.venv/bin/activate
    ```

3.  **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```
    *(You'll need to create a `requirements.txt` file using `pip freeze > requirements.txt` after installing all libraries used in your code, or list them manually).*
    **Minimum requirements:**
    ```
    pandas
    numpy
    scikit-learn
    matplotlib
    seaborn
    streamlit
    joblib
    xgboost
    lightgbm
    ```

## 4. Data Preparation and Preprocessing

The project assumes the dataset (e.g., `ai4i_df` from the AI4I 2020 Predictive Maintenance Dataset or similar structured sensor data) is available in a format like CSV.

**Key Preprocessing Steps:**

* **Feature Selection:** Identified and utilized relevant numerical features (`Air Temperature`, `Process Temperature`, `Rotational Speed`, `Torque`, `Tool Wear`). Identifier columns (`UDI`) and specific failure type flags (`TWF`, `HDF`, `PWF`, `OSF`, `RNF`) were typically excluded from the main feature set for the general failure prediction task.
* **Data Splitting:** The dataset is split into training and testing sets (e.g., 70% train, 30% test) using `sklearn.model_selection.train_test_split`. **Stratified sampling** is applied (`stratify=y`) to maintain the class distribution (especially for the minority 'failure' class) in both sets.
* **Feature Scaling:** Numerical features are scaled using `sklearn.preprocessing.StandardScaler`. This is crucial for models sensitive to feature scales (e.g., Logistic Regression, SVM, KNN) to ensure optimal convergence and performance.

## 5. Exploratory Data Analysis (EDA)

EDA was performed using various visualization techniques to gain insights into the data and the relationship between sensor readings and machine failures.

* **Time Series Plots:** To observe trends and anomalies in sensor readings leading up to failure events.
* **Box Plots / Violin Plots:** To compare the distribution of continuous sensor features between "failure" and "no failure" instances. These were instrumental in identifying shifts in sensor values associated with failure.
* **Histograms:** To visualize the frequency distribution of individual sensor readings for each failure state.
* **Correlation Heatmap:** To understand the linear relationships between all numerical features and the target variable, highlighting potentially predictive indicators.

*(Refer to the project notebooks or scripts for the detailed EDA code and visualizations.)*

## 6. Machine Learning Model Development

The problem is addressed as a binary classification task. Multiple machine learning algorithms were trained and evaluated to find the most suitable model for predicting machine failures.

### Algorithms Evaluated:

* **Logistic Regression**
* **Decision Tree Classifier**
* **Random Forest Classifier**
* **Support Vector Classifier (SVC)**
* **XGBoost Classifier**
* **LightGBM Classifier**

### Imbalance Handling:
Given that machine failures are typically a minority class, the following strategies were employed during model training to mitigate class imbalance:
* Using `class_weight='balanced'` parameter (for Logistic Regression, Decision Tree, Random Forest, SVC).
* Using `scale_pos_weight` parameter (for XGBoost, LightGBM), calculated as `(count of negative examples) / (count of positive examples)`.

## 7. Model Evaluation Metrics

For imbalanced classification problems like machine failure prediction, relying solely on accuracy can be misleading. Therefore, a comprehensive set of evaluation metrics was used:

* **Confusion Matrix:** Provides a detailed breakdown of True Positives (correctly predicted failures), True Negatives (correctly predicted non-failures), False Positives (false alarms), and False Negatives (missed failures).
* **Recall (Sensitivity / True Positive Rate):** The proportion of actual failures that were correctly identified. Crucial for minimizing missed critical events.
* **Precision (Positive Predictive Value):** The proportion of predicted failures that were actual failures. Important for reducing false alarms.
* **F1-Score:** The harmonic mean of Precision and Recall, offering a balanced measure of the model's performance.
* **ROC AUC (Receiver Operating Characteristic Area Under the Curve):** Measures the model's overall ability to distinguish between failure and non-failure states across different probability thresholds. Robust to class imbalance.
* **Balanced Accuracy:** The average of recall and specificity, providing a more balanced accuracy measure for imbalanced datasets.

## 8. How to Run the Streamlit Application

The project includes an interactive Streamlit web application to demonstrate the model's predictions.

1.  **Ensure Model and Scaler are Saved:**
    After training your Logistic Regression model and fitting your `StandardScaler`, save them as `logistic_regression_model.pkl` and `scaler.pkl` respectively. These files must be in the same directory as `app.py`.
    ```python
    import joblib
    # ... (your model training and scaler fitting code) ...
    joblib.dump(model, 'logistic_regression_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    ```

2.  **Navigate to the project directory:**
    Open your terminal or command prompt and change your current directory to where your `app.py` file is located.

3.  **Run the Streamlit application:**
    ```bash
    streamlit run app.py
    ```
    This command will open a new tab in your default web browser (usually at `http://localhost:8501`) displaying the interactive application. You can input sensor readings and get real-time predictions.

## 9. Future Enhancements

* **Hyperparameter Tuning:** Implement `GridSearchCV` or `RandomizedSearchCV` for optimal hyperparameter tuning for each model.
* **Advanced Ensemble Methods:** Explore Stacking Classifiers for potentially higher predictive performance.
* **Time-Series Analysis:** Incorporate more sophisticated time-series features (e.g., rolling averages, trends, Fourier transforms) if historical data permits.
* **Real-time Data Integration:** Connect the application to live sensor data streams for continuous monitoring.
* **Cost-Sensitive Learning:** Incorporate the actual costs of False Positives vs. False Negatives into model training or evaluation to optimize for business objectives.
* **Explainable AI (XAI):** Integrate tools like SHAP or LIME to explain individual model predictions, providing more trust and insight for maintenance personnel.
* **Dashboarding:** Enhance the Streamlit app with more visualizations (e.g., historical sensor trends, model confidence levels).

## 10. Contact

For any questions or suggestions, feel free to reach out:

Sanjay - sanjaymvkrishna@gmail.com  
Project Link: https://github.com/SANJAY-KRISHNA-MV