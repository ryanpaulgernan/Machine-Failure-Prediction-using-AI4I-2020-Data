# Machine Failure Prediction Project

## 1. Project Overview

This project aims to develop a machine learning model to predict potential machine failures or breakdowns based on sensor data. By leveraging predictive maintenance, we can enable proactive interventions, reducing downtime, optimizing maintenance costs, and enhancing operational efficiency and safety.

## 2. Data Understanding and Preprocessing

The dataset consists of various sensor readings and operational parameters, which collectively serve as **Machine Health Indicators**.

**Key Features (Machine Health Indicators) identified and used:**
* **Air Temperature [K]:** Ambient air temperature around the machine.
* **Process Temperature [K]:** Temperature of the machine's internal processes.
* **Rotational Speed [rpm]:** Revolutions per minute of the machine's rotating parts.
* **Torque [Nm]:** The rotational force being applied by the machine.
* **Tool Wear [min]:** Cumulative wear on the machine's tools.

**Target Variable:**
* **Machine Failure (Binary: 0 or 1):** A binary indicator where `0` means no failure and `1` indicates a machine failure/breakdown. This is a **classification problem**.

**Preprocessing Steps:**
* **Feature Selection:** Identified and selected numerical features relevant to machine health prediction, excluding identifiers like `UDI` and specific failure type flags like `TWF`, `HDF`, `PWF`, `OSF`, `RNF` for the primary prediction task.
* **Feature Scaling:** Applied `StandardScaler` to numerical features. This is crucial for many machine learning algorithms (e.g., Logistic Regression, SVM, KNN) to ensure they converge efficiently and perform optimally, as features with larger scales do not disproportionately influence the model.

## 3. Exploratory Data Analysis (EDA) and Visualization

Effective visualization was used to understand the relationship between sensor data and machine failures.

### Key Visualization Techniques Employed:

1.  **Time Series Plots with Failure Events:**
    * **Purpose:** To observe how sensor readings change over time and identify any patterns or anomalies leading up to a machine failure event.
    * **Methodology:** Plotted sensor values (e.g., `air_temperature`, `process_temperature`) against `timestamp` (if available), with vertical lines or shaded regions indicating `Machine failure` events.
    * **Insight:** Helps in identifying precursors to failure, such as sudden spikes, drops, or gradual trends in sensor readings before a breakdown.

2.  **Box Plots / Violin Plots (Categorical vs. Continuous):**
    * **Purpose:** To compare the distribution of continuous sensor readings for instances where the machine `fails` (1) versus when it `does not fail` (0).
    * **Methodology:** Generated box plots and violin plots for each `Machine Health Indicator` (continuous) grouped by the `Machine failure` (binary) status.
    * **Insight:** Reveals differences in median, quartiles, spread, and overall data density between the two failure states. For instance, a clear shift in the median temperature when a failure occurs indicates a strong relationship. Violin plots were particularly useful for showing the full density distribution, including potential multi-modal patterns.

3.  **Histograms (Overlayed/Stacked):**
    * **Purpose:** To visualize the frequency distribution of sensor readings for each failure state.
    * **Methodology:** Plotted histograms or Kernel Density Estimates (KDEs) for continuous features, separating or overlaying the distributions for `Machine failure = 0` and `Machine failure = 1`.
    * **Insight:** Provides a granular view of where data points cluster within each group, complementing box/violin plots.

4.  **Correlation Heatmap:**
    * **Purpose:** To visualize the pairwise correlation coefficients between all numerical `Machine Health Indicators` and the `Machine failure` target.
    * **Methodology:** Computed the correlation matrix of the preprocessed numerical features and plotted it using `seaborn.heatmap` with annotations and a diverging colormap (`coolwarm`).
    * **Insight:** Identifies strong positive or negative linear relationships between features, and importantly, between features and the `Machine failure` target, indicating which features are most indicative of failure.

## 4. Machine Learning Model Development

The problem was framed as a **binary classification** task. A range of classification algorithms were considered and set up for comparative evaluation.

### Algorithms Selected for Comparison:

* **Logistic Regression:** A linear, interpretable baseline model.
* **Decision Tree Classifier:** A tree-based model capable of capturing non-linear relationships.
* **Random Forest Classifier:** An ensemble method (bagging) known for its accuracy and robustness to overfitting.
* **Support Vector Classifier (SVC):** A powerful model that finds optimal hyperplanes for classification, effective in high-dimensional spaces.
* **XGBoost Classifier:** A highly optimized Gradient Boosting Machine (GBM) known for its speed and top-tier performance.
* **LightGBM Classifier:** Another fast and efficient GBM, particularly good for large datasets.
* **(Optional) CatBoost Classifier:** A GBM that handles categorical features automatically and is robust to noise.
* **(Optional) K-Nearest Neighbors (KNN):** A simple, non-parametric algorithm based on local proximity.

### Model Training and Evaluation Methodology:

1.  **Data Splitting:** The dataset was split into training and testing sets (e.g., 70% train, 30% test). **Stratified sampling** was used during splitting to ensure that the proportion of `Machine failure` (minority class) was maintained in both training and testing sets, which is crucial for imbalanced datasets.
2.  **Class Imbalance Handling:** Since machine failures are typically rare (imbalanced dataset), the following strategies were incorporated into model training:
    * `class_weight='balanced'` parameter for `LogisticRegression`, `DecisionTreeClassifier`, `RandomForestClassifier`, and `SVC`.
    * `scale_pos_weight` parameter for `XGBoost` and `LightGBM`, calculated as `(count of negative examples) / (count of positive examples)`.
    * (For CatBoost) `class_weights` parameter can be used.
3.  **Iterative Training and Evaluation:** A loop was implemented to systematically:
    * Instantiate each model with appropriate parameters (including `random_state` for reproducibility).
    * Fit the model on the `scaled training data`.
    * Make predictions on the `scaled test data`.
    * Calculate and print a comprehensive set of evaluation metrics.

### Key Evaluation Metrics Employed (Crucial for Imbalanced Classification):

* **Confusion Matrix:** Provides a breakdown of True Positives (TP), True Negatives (TN), False Positives (FP), and False Negatives (FN). This is the foundation for all other metrics.
* **Recall (Sensitivity / True Positive Rate):**
    * **Focus:** Minimizing **False Negatives (FN)** – missing actual failures. Critical for machine failure prediction as missing a failure can be very costly.
    * **Formula:** $TP / (TP + FN)$
* **Precision (Positive Predictive Value):**
    * **Focus:** Minimizing **False Positives (FP)** – false alarms. Important to avoid unnecessary maintenance or operational disruptions.
    * **Formula:** $TP / (TP + FP)$
* **F1-Score:**
    * **Focus:** Harmonic mean of Precision and Recall, providing a balanced measure.
* **ROC AUC (Receiver Operating Characteristic Area Under the Curve):**
    * **Focus:** Overall ability of the model to distinguish between positive and negative classes across all classification thresholds. Robust to class imbalance.
* **Balanced Accuracy:**
    * **Focus:** The average of recall and specificity, providing a more reliable accuracy measure for imbalanced datasets than raw accuracy.

## 5. Model Deployment (Streamlit Application)

To make the model interactive and accessible, a simple web application was developed using Streamlit.

### Streamlit Application Features:

* **Model and Scaler Loading:** The trained Logistic Regression model (`logistic_regression_model.pkl`) and the fitted `StandardScaler` (`scaler.pkl`) are loaded at startup, ensuring consistent predictions.
* **Interactive Input Fields:** Users can input real-time or hypothetical sensor readings (`Air Temperature`, `Process Temperature`, `Rotational Speed`, `Torque`) via numerical input widgets.
* **Prediction Trigger:** A "Predict Machine Failure Risk" button triggers the prediction process.
* **Real-time Prediction Display:**
    * The application scales the user input using the loaded `StandardScaler`.
    * It then uses the loaded model to predict the machine's failure status (0 or 1).
    * It also displays the predicted probability of failure and no failure.
    * Clear visual feedback (e.g., "🔴 HIGH RISK OF FAILURE!" or "🟢 Machine is operating normally.") is provided.

### Running the Streamlit App:

1.  Ensure you have `streamlit`, `scikit-learn`, and `joblib` installed (`pip install streamlit scikit-learn joblib`).
2.  Save your trained model and scaler (e.g., `logistic_regression_model.pkl` and `scaler.pkl`) in the same directory as your Streamlit app file (`app.py`).
3.  Run the app from your terminal: `streamlit run app.py`

This comprehensive documentation covers all the major aspects of your machine failure prediction project, from data to deployment.