import pandas as pd
from pathlib import Path

RAW_PATH = "data/raw/Admission.csv"
PROCESSED_PATH = "data/processed/admission_clean.csv"


def main():

    df = pd.read_csv(RAW_PATH)

    # remove ID column
    if "Serial_No" in df.columns:
        df = df.drop(columns=["Serial_No"])

    # convert probability to binary classification
    df["Admit_Chance"] = (df["Admit_Chance"] >= 0.8).astype(int)

    # Create processed folder if it does not exist
    Path("data/processed").mkdir(parents=True, exist_ok=True)

    # Save cleaned dataset
    df.to_csv(PROCESSED_PATH, index=False)


if __name__ == "__main__":
    main()
