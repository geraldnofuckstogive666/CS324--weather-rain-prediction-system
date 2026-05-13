import streamlit as st
import pandas as pd
import numpy as np
import joblib

from pathlib import Path


# =========================================
# PAGE CONFIG
# =========================================

st.set_page_config(
    page_title="Weather Rain Prediction System",
    page_icon="🌧️",
    layout="wide"
)


# =========================================
# LOAD MODEL
# =========================================

model = joblib.load(
    Path("models/random_forest_model.pkl")
)


# =========================================
# SIDEBAR
# =========================================

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Go To",
    [
        "Dashboard",
        "Prediction"
    ]
)


# =========================================
# DASHBOARD PAGE
# =========================================

if page == "Dashboard":

    st.title("Weather Rain Prediction System")

    st.markdown("---")

    st.header("Project Overview")

    st.write("""
    This machine learning project predicts whether rain will occur
    based on weather-related input features.

    The system was trained using:
    - Logistic Regression
    - Decision Tree
    - Random Forest

    The best-performing model is the Random Forest Classifier.
    """)

    st.markdown("---")

    st.header("📊 Dataset Information")

    col1, col2, col3 = st.columns(3)

    col1.metric("Dataset Rows", "14,471")
    col2.metric("Features Used", "18")
    col3.metric("Best Accuracy", "90.64%")

    st.markdown("---")

    st.header("Models Used")

    models_df = pd.DataFrame({
        "Model": [
            "Logistic Regression",
            "Decision Tree",
            "Random Forest"
        ],
        "Validation Accuracy": [
            "76.95%",
            "79.65%",
            "90.46%"
        ],
        "Rain Recall": [
            "87%",
            "78%",
            "33%"
        ]
    })

    st.dataframe(
        models_df,
        use_container_width=True
    )

    st.markdown("---")

    st.header("🏆 Best Model")

    st.success("""
    Random Forest achieved the best overall performance
    with strong generalization and stable testing accuracy.
    """)

    st.markdown("---")

    st.header("Evaluation Metrics Used")

    st.write("""
    - Accuracy → Overall correctness of predictions
    - Precision → Correctness of rain predictions
    - Recall → Ability to detect actual rain cases
    - F1-Score → Balance between precision and recall
    - Confusion Matrix → Detailed prediction breakdown
    """)


# =========================================
# PREDICTION PAGE
# =========================================

elif page == "Prediction":

    st.title("🌦️ Rain Prediction")

    st.write("""
    Enter weather conditions below to predict
    whether rain is likely to occur.
    """)

    st.markdown("---")

    # =====================================
    # USER INPUTS
    # =====================================

    col1, col2 = st.columns(2)

    with col1:

        visibility = st.number_input(
            "Visibility",
            value=10000.0
        )

        temperature = st.number_input(
            "Temperature",
            value=29.0
        )

        feels_like = st.number_input(
            "Feels Like Temperature",
            value=32.0
        )

        temp_min = st.number_input(
            "Minimum Temperature",
            value=28.0
        )

        temp_max = st.number_input(
            "Maximum Temperature",
            value=31.0
        )

        pressure = st.number_input(
            "Pressure",
            value=1012.0
        )

    with col2:

        humidity = st.slider(
            "Humidity",
            0,
            100,
            85
        )

        wind_speed = st.number_input(
            "Wind Speed",
            value=2.0
        )

        wind_degree = st.number_input(
            "Wind Degree",
            value=180.0
        )

        wind_gust = st.number_input(
            "Wind Gust",
            value=4.0
        )

        cloudiness = st.slider(
            "Cloud Coverage",
            0,
            100,
            75
        )

    st.markdown("---")

    # =====================================
    # PREDICT BUTTON
    # =====================================

    if st.button("Predict Rain"):

        # =================================
        # HIDDEN DEFAULT VALUES
        # =================================

        coord_lon = 120.98
        coord_lat = 14.60

        sea_level = 1012
        ground_level = 1008

        city_name = 0
        sys_type = 1
        sys_id = 1015019

        # =================================
        # CREATE INPUT ARRAY
        # =================================

        input_data = np.array([[
            visibility,
            coord_lon,
            coord_lat,
            temperature,
            feels_like,
            temp_min,
            temp_max,
            pressure,
            humidity,
            sea_level,
            ground_level,
            wind_speed,
            wind_degree,
            wind_gust,
            cloudiness,
            city_name,
            sys_type,
            sys_id
        ]])

        # =================================
        # PREDICTION
        # =================================

        prediction = model.predict(input_data)

        st.markdown("---")

        st.header("📢 Prediction Result")

        if prediction[0] == 1:

            st.error("🌧️ Rain Expected")

        else:

            st.success("☀️ No Rain Expected")

        st.markdown("---")

        st.subheader("Input Summary")

        summary_df = pd.DataFrame({
            "Feature": [
                "Temperature",
                "Humidity",
                "Pressure",
                "Wind Speed",
                "Cloud Coverage"
            ],
            "Value": [
                temperature,
                humidity,
                pressure,
                wind_speed,
                cloudiness
            ]
        })

        st.dataframe(
            summary_df,
            use_container_width=True
        )