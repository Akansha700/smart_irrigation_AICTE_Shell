import streamlit as st
import numpy as np
import pandas as pd
import joblib
import datetime

# Load the trained model
model = joblib.load("Farm_Irrigation_System.pkl")

# Title and instructions
st.title("💧 Smart Sprinkler System")
st.subheader("🌱 Predict which sprinklers should be ON or OFF based on sensor input (scaled 0 to 1)")

# Sidebar info
st.sidebar.title("ℹ️ About")
st.sidebar.markdown("""
This app takes 20 sensor readings (scaled 0.0 to 1.0) and uses a trained machine learning model
to predict which sprinklers in a field should be activated.
""")

# Collect sensor values
sensor_values = []
cols = st.columns(4)  # 4-column layout

for i in range(20):
    col = cols[i % 4]
    val = col.slider(f"Sensor {i}", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
    sensor_values.append(val)

# Optional: Show bar chart of sensor inputs
st.markdown("### 📊 Sensor Input Overview")
df_input = pd.DataFrame({'Sensor': [f'Sensor {i}' for i in range(20)], 'Value': sensor_values})
st.bar_chart(df_input.set_index('Sensor'))

# Predict button
if st.button("🚀 Predict Sprinklers"):
    input_array = np.array(sensor_values).reshape(1, -1)
    prediction = model.predict(input_array)[0]

    # Make sure prediction is iterable
    if not isinstance(prediction, (list, np.ndarray)):
        prediction = [prediction]

    num_sprinklers = len(prediction)

    st.markdown("### 🌿 Sprinkler Status Prediction:")

    result_cols = st.columns(4)
    for i in range(num_sprinklers):
        col = result_cols[i % 4]
        with col:
            if prediction[i] == 1:
                st.success(f"💧 Sprinkler {i}: ON")
            else:
                st.error(f"🌾 Sprinkler {i}: OFF")

    # Save prediction + inputs to CSV log
    all_data = sensor_values + list(prediction)
    column_names = [f"Sensor_{i}" for i in range(len(sensor_values))] + [f"Sprinkler_{i}" for i in range(num_sprinklers)]

    log_data = pd.DataFrame([all_data], columns=column_names)
    log_data['Timestamp'] = datetime.datetime.now()

    try:
        log_data.to_csv("sprinkler_predictions_log.csv", mode='a', header=False, index=False)
    except FileNotFoundError:
        log_data.to_csv("sprinkler_predictions_log.csv", mode='w', header=True, index=False)

    st.success("📁 Prediction saved to log file!")

# Footer
st.markdown("---")
st.caption("🚀 Built with Streamlit | 👩‍💻 Developed by Your Name")
