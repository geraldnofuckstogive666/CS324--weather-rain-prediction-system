import pandas as pd
import numpy as np

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler


def load_and_preprocess_data():

    # ==============================
    # LOAD DATASET
    # ==============================

    DATA_PATH = Path("Datasets/202311_CombinedData.csv")

    df = pd.read_csv(DATA_PATH)

    print("Dataset Loaded Successfully!")

    # ==============================
    # REMOVE UNUSED COLUMNS
    # ==============================
    unused_columns = [
        "datetime",
        "weather.description",
        "weather.icon",
        "sys.sunrise",
        "sys.sunset",
        "extraction_date_time",
        "weather.main",
        "weather.id"
    ]

    df = df.drop(columns=unused_columns, errors='ignore')

    print("Unused columns removed!")

    # ==============================
    # HANDLE MISSING VALUES
    # ==============================


    df['rain.1h'] = pd.to_numeric(df['rain.1h'], errors='coerce')

    df['rain.1h'] = df['rain.1h'].fillna(0)

    numeric_columns = df.select_dtypes(include=np.number).columns

    for col in numeric_columns:

        if col != 'rain.1h':
            df[col] = df[col].fillna(df[col].mean())

    categorical_columns = df.select_dtypes(
        include=['object', 'string']
    ).columns

    for col in categorical_columns:
        df[col] = df[col].fillna(df[col].mode()[0])

    print("Missing values handled!")

    # ==============================
    # CREATE TARGET VARIABLE
    # ==============================

    df['WillRain'] = df['rain.1h'].apply(
        lambda x: 1 if float(x) > 0 else 0
    )

    print("Target variable created!")

    # ==============================
    # REMOVE TARGET LEAK COLUMN
    # ==============================
    df = df.drop(columns=['rain.1h'])

    print("Target leakage removed!")

    # ==============================
    # ENCODE CATEGORICAL COLUMNS
    # ==============================

    label_encoders = {}

    categorical_columns = df.select_dtypes(
        include=['object', 'string']
    ).columns

    for col in categorical_columns:

        encoder = LabelEncoder()

        df[col] = encoder.fit_transform(df[col])

        label_encoders[col] = encoder

    print("Categorical columns encoded!")

    # ==============================
    # SAVE CLEANED DATASET
    # ==============================

    cleaned_path = Path("Datasets/cleaned_weather_data.csv")

    df.to_csv(cleaned_path, index=False)

    print("Cleaned dataset saved!")

    # ==============================
    # SPLIT FEATURES/TARGET
    # ==============================

    X = df.drop(columns=['WillRain'])
    
    print(X.columns.tolist())

    y = df['WillRain']
    # ==============================
    # TRAIN / VALIDATION / TEST SPLIT
    # ==============================

    X_remaining, X_test, y_remaining, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    X_train, X_validation, y_train, y_validation = train_test_split(
        X_remaining,
        y_remaining,
        test_size=0.25,
        random_state=42,
        stratify=y_remaining
    )

    print("Data splitting completed!")

    # ==============================
    # FEATURE SCALING
    # ==============================

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)

    X_validation_scaled = scaler.transform(X_validation)

    X_test_scaled = scaler.transform(X_test)

    print("Feature scaling completed!")
    return (
        X_train,
        X_validation,
        X_test,
        y_train,
        y_validation,
        y_test,
        X_train_scaled,
        X_validation_scaled,
        X_test_scaled,
        scaler
    )