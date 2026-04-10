import pandas as pd
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

RAW_PATH = "data/raw/Admission.csv"
PROCESSED_PATH = "data/processed/admission_clean.csv"


def main():
    try:
        logger.info("Loading raw data")

        df = pd.read_csv(RAW_PATH)

        if "Serial_No" in df.columns:
            df = df.drop(columns=["Serial_No"])

        df["Admit_Chance"] = (df["Admit_Chance"] >= 0.8).astype(int)

        Path("data/processed").mkdir(parents=True, exist_ok=True)

        df.to_csv(PROCESSED_PATH, index=False)
        logger.info("Cleaned dataset saved")

    except Exception as e:
        logger.error(f"make_dataset error: {e}")
        raise


if __name__ == "__main__":
    main()
