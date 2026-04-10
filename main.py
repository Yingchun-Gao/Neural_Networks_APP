from src.data.make_dataset import main as make_dataset
from src.features.build_features import build_features
from src.models.train_model import (
    train_model,
    evaluate_model,
    save_model,
    save_scaler,
    save_feature_columns,
)
from src.visualization.visualize import plot_correlation, plot_loss_curve
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_pipeline():
    try:
        logger.info("Pipeline started")

        make_dataset()

        X_train, X_test, y_train, y_test, scaler = build_features()

        model = train_model(X_train, y_train)

        accuracy, matrix = evaluate_model(model, X_test, y_test)

        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"Confusion Matrix:\n{matrix}")

        save_model(model)
        save_scaler(scaler)

        df = pd.read_csv("data/processed/admission_clean.csv")
        df = pd.get_dummies(df, columns=["University_Rating", "Research"], dtype=int)

        feature_columns = df.drop("Admit_Chance", axis=1).columns
        save_feature_columns(feature_columns)

        plot_correlation(df)
        plot_loss_curve(model)

        logger.info("Pipeline finished")

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    run_pipeline()
