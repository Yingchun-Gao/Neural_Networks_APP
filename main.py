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


def run_pipeline():

    # Step 1: Prepare dataset
    make_dataset()

    # Step 2: Build features
    X_train, X_test, y_train, y_test, scaler = build_features()

    # Step 3: Train model
    model = train_model(X_train, y_train)

    # Step 4: Evaluate model
    accuracy, matrix = evaluate_model(model, X_test, y_test)

    print(f"Test Accuracy: {accuracy:.4f}")
    print(matrix)

    # Step 5: Save model artifacts
    save_model(model)
    save_scaler(scaler)

    # Step 6: Save feature column structure
    df = pd.read_csv("data/processed/admission_clean.csv")
    df = pd.get_dummies(df, columns=["University_Rating", "Research"], dtype=int)

    feature_columns = df.drop("Admit_Chance", axis=1).columns
    save_feature_columns(feature_columns)

    # Step 7: Visualizations
    plot_correlation(df)
    plot_loss_curve(model)


if __name__ == "__main__":
    run_pipeline()
