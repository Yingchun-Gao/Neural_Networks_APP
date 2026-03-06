from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import pickle
import os


def train_model(X_train, y_train):

    model = MLPClassifier(
        hidden_layer_sizes=(3,),
        activation="tanh",
        batch_size=50,
        max_iter=300,
        random_state=123,
    )

    model.fit(X_train, y_train)

    return model


def evaluate_model(model, X_test, y_test):

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    matrix = confusion_matrix(y_test, predictions)

    return accuracy, matrix


def save_model(model):

    os.makedirs("models", exist_ok=True)

    with open("models/mlp_model.pkl", "wb") as f:
        pickle.dump(model, f)


def save_scaler(scaler):

    with open("models/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)


def save_feature_columns(columns):

    with open("models/feature_columns.pkl", "wb") as f:
        pickle.dump(columns, f)
