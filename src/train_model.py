import numpy as np

from sklearn.utils.class_weight import compute_class_weight

from sklearn.model_selection import (
    train_test_split,
    RandomizedSearchCV
)

from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier
)

from xgboost import XGBClassifier

def split_data(X, y):

    return train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )
def get_class_weights(y_train):

    classes = np.unique(y_train)

    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train
    )

    return dict(zip(classes, weights))
def train_gradient_boosting(
    X_train,
    y_train
):

    gb = GradientBoostingClassifier()

    params = {
        "n_estimators": [100, 200],
        "learning_rate": [0.01, 0.1, 0.2],
        "max_depth": [3, 5, 7]
    }

    search = RandomizedSearchCV(
        gb,
        params,
        n_iter=15,
        cv=3,
        scoring="accuracy",
        n_jobs=-1
    )

    search.fit(X_train, y_train)

    return search.best_estimator_