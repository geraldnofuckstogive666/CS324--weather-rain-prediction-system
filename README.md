# CS324--weather-rain-prediction-system
A public repository for CS324 - Machine Learning Final Project 


# Weather Rain Prediction System

A Machine Learning-based Weather Rain Prediction System developed using Python, Scikit-learn, and Streamlit.

The project predicts whether rain is expected based on weather-related features such as:

- Temperature
- Humidity
- Pressure
- Wind Speed
- Wind Gust
- Cloud Coverage
- Visibility

---

# Machine Learning Models Used

The system was trained and evaluated using:

1. Logistic Regression
2. Decision Tree Classifier
3. Random Forest Classifier

The Random Forest model achieved the best overall performance and was selected as the final deployed model.

---

# Project Structure

```bash
WeatherRainPrediction/
│
├── Datasets/
│   ├── 202311_CombinedData.csv
│   └── cleaned_weather_data.csv
│
├── models/
│   ├── random_forest_model.pkl
│   └── scaler.pkl
│
├── src/
│   ├── preprocessing.py
│   ├── train_models.py
│   ├── evaluate_models.py
│   └── predict.py
│
├── ui/
│   └── app.py
│
├── main.py
├── requirements.txt
└── README.md

```bash

## Dataset Information
Dataset Rows: 14,471
Features Used: 18
Target Variable:
1 = Rain Expected
0 = No Rain

##Machine Learning Pipeline

The project follows the standard machine learning workflow:

  Data Collection
  Data Cleaning
  Missing Value Handling
  Feature Encoding
  Feature Scaling
  Data Splitting
  Model Training
  Model Evaluation
  Model Deployment

##Evaluation Metrics Used

The following metrics were used to evaluate the models:
  
  Accuracy
  Precision
  Recall
  F1-Score
  Confusion Matrix

## Final Model Performance
#### Random Forest Final Test Results
Accuracy: ~90%
Rain Detection Recall: ~58%
Balanced Performance Across Classes


## Installation Guide
### 1. Clone Repository
    
    git clone <https://github.com/geraldnofuckstogive666/CS324--weather-rain-prediction-system.git>

### 2. Navigate Into Project

  cd WeatherRainPrediction

### 3. Install Required Dependencies

  pip install -r requirements.txt

## Run The Machine Learning Pipeline

Train models and generate saved model files:

  py main.py

## Run Streamlit Web Application

    streamlit run ui/app.py


## Streamlit Deployment

This project can also be deployed using Streamlit Cloud.

### Deployment Steps
Push project to GitHub
Open Streamlit Cloud
Connect GitHub repository
Select:
ui/app.py
Deploy application
Dependencies

Main libraries used:

pandas
numpy
scikit-learn
streamlit
joblib
