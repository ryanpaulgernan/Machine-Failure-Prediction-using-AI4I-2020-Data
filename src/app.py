import streamlit as st
import joblib
import numpy as np
import pandas as pd

# --- Load the Trained Model and Scaler ---
# Make sure these paths are correct, relative to where your app.py is located
try:
    model_path = r'E:\Machine-Failure-Prediction-using-AI4I-2020-Data\models\logistic_regression_model.pkl'
    model = joblib.load(model_path)
    # scaler = joblib.load('scaler.pkl')
except FileNotFoundError:
    st.error("Error: Model or scaler file not found. "
             "Please ensure 'logistic_regression_model.pkl' and 'scaler.pkl' "
             "are in the same directory as this app.py file.")
    st.stop() # Stop the app if files aren't found


# --- Streamlit UI ---

st.set_page_config(page_title="Machine Failure Prediction", layout="centered")

st.title("⚙️ Machine Failure Prediction")
st.write("Enter the sensor readings below to predict if the machine is at risk of failure.")

# Input fields for sensor readings
st.header("Sensor Readings")

air_temperature = st.number_input(
    "Air Temperature [K]",
    min_value=290.0,
    max_value=310.0, # Adjust based on your data's realistic range
    value=300.0,
    step=0.1,
    help="Typical range might be around 295K to 305K."
)

process_temperature = st.number_input(
    "Process Temperature [K]",
    min_value=300.0,
    max_value=320.0, # Adjust based on your data's realistic range
    value=310.0,
    step=0.1,
    help="Often higher than air temperature. Typical range 305K to 315K."
)

rotational_speed = st.number_input(
    "Rotational Speed [rpm]",
    min_value=1000.0,
    max_value=2000.0, # Adjust based on your data's realistic range
    value=1500.0,
    step=1.0,
    help="Typical range might be around 1300rpm to 1600rpm."
)

torque = st.number_input(
    "Torque [Nm]",
    min_value=0.0,
    max_value=100.0, # Adjust based on your data's realistic range
    value=50.0,
    step=0.1,
    help="Typical range might be around 40Nm to 65Nm."
)

# Tool wear (if it was a numerical feature used in your model)
# If your model used 'tool_wear', uncomment and adjust
tool_wear = st.number_input(
    "Tool Wear [min]",
    min_value=0.0,
    max_value=250.0, # Adjust based on your data's realistic range
    value=50.0,
    step=1.0,
    help="Cumulative tool wear in minutes."
)

# --- Prediction Button ---
if st.button("Predict Machine Failure Risk"):
    # Prepare input data for prediction
    # Ensure the order of features matches the order used during model training
    input_data = pd.DataFrame([[
        air_temperature,
        process_temperature,
        rotational_speed,
        torque,
        tool_wear
        # Add other features like 'tool_wear' here if used in your model
    ]], columns=['Air temperature [K]','Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]','Tool wear [min]']) # Ensure column names match trained model's features

    # Scale the input data using the loaded scaler
    # scaled_input_data = scaler.transform(input_data)

    # Make prediction
    prediction = model.predict(input_data)
    prediction_proba = model.predict_proba(input_data)[0] # Get probabilities for the first (and only) input

    st.subheader("Prediction Result:")

    if prediction[0] == 1:
        st.error(f"🔴 **HIGH RISK OF FAILURE!**")
        st.write(f"Probability of Failure: **{prediction_proba[1]*100:.2f}%**")
        st.write(f"Probability of No Failure: {prediction_proba[0]*100:.2f}%")
        st.warning("Immediate attention or maintenance recommended.")
    else:
        st.success(f"🟢 **Machine is operating normally.**")
        st.write(f"Probability of No Failure: **{prediction_proba[0]*100:.2f}%**")
        st.write(f"Probability of Failure: {prediction_proba[1]*100:.2f}%")
        st.info("Continue monitoring sensor readings regularly.")

st.markdown("---")
st.markdown("This application is for demonstration purposes. Always consult with maintenance professionals for critical decisions.")
st.markdown("Developed with ❤️ using Streamlit and scikit-learn.")