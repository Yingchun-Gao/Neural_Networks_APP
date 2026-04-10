import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATA_PATH = "data/processed/admission_clean.csv"


def build_features():
    try:
        logger.info("Building features")

        df = pd.read_csv(DATA_PATH)

        df = pd.get_dummies(df, columns=["University_Rating", "Research"], dtype=int)

        X = df.drop("Admit_Chance", axis=1)
        y = df["Admit_Chance"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=123, stratify=y
        )

        scaler = MinMaxScaler()

        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        return X_train_scaled, X_test_scaled, y_train, y_test, scaler

    except Exception as e:
        logger.error(f"build_features error: {e}")
        raise
