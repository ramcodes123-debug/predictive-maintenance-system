import streamlit as st
import sys
from pathlib import Path

# =========================================================
# PROJECT ROOT
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import prediction function
from SRC.predict import predict


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Predictive Maintenance System",
    page_icon="⚙️",
    layout="centered"
)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header("⚙️ Predictive Maintenance")

    st.write(
        """
        This application uses a Machine Learning model
        to predict whether a machine is likely to fail
        based on sensor readings.
        """
    )

    st.divider()

    st.subheader("Model Information")

    st.write("**Algorithm:** Random Forest")
    st.write("**Task:** Binary Classification")
    st.write("**Dataset:** AI4I 2020 Predictive Maintenance")
    st.write("**Target:** Machine Failure")

    st.divider()

    st.subheader("Input Features")

    st.write(
        """
        • Machine Type  
        • Air Temperature  
        • Process Temperature  
        • Rotational Speed  
        • Torque  
        • Tool Wear
        """
    )


# =========================================================
# TITLE
# =========================================================

st.title("⚙️ Predictive Maintenance System")

st.write(
    "Enter the machine sensor readings below to predict "
    "the probability of machine failure."
)


# =========================================================
# MACHINE INFORMATION
# =========================================================

st.subheader("🏭 Machine Information")

machine_type = st.selectbox(
    "Machine Type",
    options=["L", "M", "H"],
    index=1,
    help="L = Low, M = Medium, H = High"
)


# =========================================================
# SENSOR READINGS
# =========================================================

st.subheader("📊 Sensor Readings")

col1, col2 = st.columns(2)

with col1:

    air_temp = st.number_input(
        "Air Temperature (K)",
        min_value=290.0,
        max_value=310.0,
        value=300.0,
        step=0.1
    )

    rot_speed = st.number_input(
        "Rotational Speed (rpm)",
        min_value=500,
        max_value=3000,
        value=1500,
        step=10
    )

    tool_wear = st.number_input(
        "Tool Wear (min)",
        min_value=0,
        max_value=300,
        value=100,
        step=1
    )


with col2:

    process_temp = st.number_input(
        "Process Temperature (K)",
        min_value=300.0,
        max_value=320.0,
        value=310.0,
        step=0.1
    )

    torque = st.number_input(
        "Torque (Nm)",
        min_value=0.0,
        max_value=100.0,
        value=40.0,
        step=0.1
    )


# =========================================================
# PREDICTION BUTTON
# =========================================================

st.divider()

predict_button = st.button(
    "🔮 Predict Machine Failure",
    use_container_width=True
)


# =========================================================
# PREDICTION
# =========================================================

if predict_button:

    raw_input = {
        "air_temp": air_temp,
        "process_temp": process_temp,
        "rot_speed": rot_speed,
        "torque": torque,
        "tool_wear": tool_wear,
        "type": machine_type,
    }

    try:

        result = predict(raw_input)

        prediction = result["prediction"]
        probability = result["failure_probability"]

        # Convert probability to percentage
        probability_percent = probability * 100


        # =================================================
        # RESULT
        # =================================================

        st.divider()

        st.subheader("🔍 Prediction Result")


        if prediction == 1:

            st.error(
                "🔴 MACHINE FAILURE PREDICTED"
            )

            risk_level = "High Risk"

        else:

            st.success(
                "🟢 NO FAILURE PREDICTED"
            )

            risk_level = "Low Risk"


        # =================================================
        # METRICS
        # =================================================

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Failure Probability",
                f"{probability_percent:.2f}%"
            )

        with col2:

            st.metric(
                "Risk Level",
                risk_level
            )


        # =================================================
        # PROBABILITY BAR
        # =================================================

        st.write("### Failure Probability")

        st.progress(
            min(probability, 1.0)
        )


        # =================================================
        # EXPLANATION
        # =================================================

        st.write("### 📋 Prediction Explanation")

        if prediction == 1:

            st.warning(
                """
                The machine's current sensor readings indicate
                an increased likelihood of failure.

                Consider inspecting the machine and checking
                components associated with temperature, torque,
                rotational speed, and tool wear.
                """
            )

        else:

            st.info(
                """
                The machine's current sensor readings indicate
                a low probability of failure.

                The model does not identify the current readings
                as strongly associated with machine failure.
                Continue normal monitoring and maintenance.
                """
            )


        # =================================================
        # INPUT SUMMARY
        # =================================================

        st.write("### 📊 Input Summary")

        input_data = {
            "Machine Type": machine_type,
            "Air Temperature": f"{air_temp:.2f} K",
            "Process Temperature": f"{process_temp:.2f} K",
            "Rotational Speed": f"{rot_speed} rpm",
            "Torque": f"{torque:.2f} Nm",
            "Tool Wear": f"{tool_wear} min"
        }

        st.table(input_data)


    except Exception as e:

        st.error(
            f"Prediction failed: {e}"
        )