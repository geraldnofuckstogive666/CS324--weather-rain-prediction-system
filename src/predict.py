import joblib
import numpy as np

from pathlib import Path


# LOAD SAVED MODEL
model = joblib.load(
    Path("models/random_forest_model.pkl")
)


# EXAMPLE MANUAL INPUT
# Replace these with real values later

sample_input = np.array([
    [
        10000,
        120.98,
        14.60,
        29,
        85,
        1012,
        2.1,
        180,
        4,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0
    ]
])


prediction = model.predict(sample_input)


if prediction[0] == 1:
    print("Prediction: Rain")
else:
    print("Prediction: No Rain")