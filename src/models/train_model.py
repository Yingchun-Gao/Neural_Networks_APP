from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import pickle
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_model(X_train, y_train):
    try:
        logger.info("Training model")

        model = MLPClassifier(
            hidden_layer_sizes=(3,),
            activation="tanh",
            batch_size=50,
            max_iter=300,
            random_state=123,
        )

        model.fit(X_train, y_train)

        return model

    except Exception as e:
        logger.error(f"train_model error: {e}")
        raise


def evaluate_model(model, X_test, y_test):
    try:
        predictions = model.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)
        matrix = confusion_matrix(y_test, predictions)

        return accuracy, matrix

    except Exception as e:
        logger.error(f"evaluate_model error: {e}")
        raise


def save_model(model):
    try:
        os.makedirs("models", exist_ok=True)

        with open("models/mlp_model.pkl", "wb") as f:
            pickle.dump(model, f)

        logger.info("Model saved")

    except Exception as e:
        logger.error(f"save_model error: {e}")
        raise


def save_scaler(scaler):
    try:
        with open("models/scaler.pkl", "wb") as f:
            pickle.dump(scaler, f)

        logger.info("Scaler saved")

    except Exception as e:
        logger.error(f"save_scaler error: {e}")
        raise


def save_feature_columns(columns):
    try:
        with open("models/feature_columns.pkl", "wb") as f:
            pickle.dump(columns, f)

        logger.info("Feature columns saved")

    except Exception as e:
        logger.error(f"save_feature_columns error: {e}")
        raise
