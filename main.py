import joblib

from pathlib import Path

from src.preprocessing import load_and_preprocess_data

from src.train_models import (
    train_logistic_regression,
    train_decision_tree,
    train_random_forest
)

from src.evaluate_models import evaluate_model

# ==============================
# LOAD + PREPROCESS DATA
# ==============================
(
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

) = load_and_preprocess_data()


# ==============================
# TRAIN MODELS
# ==============================

logistic_model = train_logistic_regression(
    X_train_scaled,
    y_train
)

print("\nLogistic Regression model trained!")


decision_tree_model = train_decision_tree(
    X_train,
    y_train
)

print("\nDecision Tree model trained!")


random_forest_model = train_random_forest(
    X_train,
    y_train
)

print("\nRandom Forest model trained!")


# ==============================
# VALIDATION EVALUATION
# ==============================

print("\n========== VALIDATION RESULTS ==========")


evaluate_model(
    "Logistic Regression",
    logistic_model,
    X_validation_scaled,
    y_validation
)


evaluate_model(
    "Decision Tree",
    decision_tree_model,
    X_validation,
    y_validation
)


evaluate_model(
    "Random Forest",
    random_forest_model,
    X_validation,
    y_validation
)
# ==============================
# FINAL TEST EVALUATION
# ==============================

print("\n========== FINAL TEST RESULTS ==========")


evaluate_model(
    "Random Forest Final Test",
    random_forest_model,
    X_test,
    y_test
)


# ==============================
# SAVE MODEL + SCALER
# ==============================

models_path = Path("models")

models_path.mkdir(exist_ok=True)

joblib.dump(
    random_forest_model,
    models_path / "random_forest_model.pkl"
)

joblib.dump(
    scaler,
    models_path / "scaler.pkl"
)

print("\nModel and scaler saved successfully!")