from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


def evaluate_model(model_name, model, X_data, y_data):
    predictions = model.predict(X_data)
    accuracy = accuracy_score(y_data, predictions)

    print(f"\n{model_name} Accuracy: {accuracy:.4f}")

    cm = confusion_matrix(y_data, predictions)

    print(f"\n{model_name} Confusion Matrix:")
    print(cm)

    report = classification_report(y_data, predictions)

    print(f"\n{model_name} Classification Report:")
    print(report)