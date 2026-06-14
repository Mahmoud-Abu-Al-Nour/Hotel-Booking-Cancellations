import joblib

from sklearn.metrics import (
    accuracy_score
)

def save_model(
    model,
    path
):

    joblib.dump(
        model,
        path
    )
    
def load_model(path):

    return joblib.load(path)

def evaluate_model(
    model,
    X_test,
    y_test
):

    y_pred = model.predict(X_test)

    return accuracy_score(
        y_test,
        y_pred
    )
    