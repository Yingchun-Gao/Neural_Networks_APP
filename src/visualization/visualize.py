import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

os.makedirs("models", exist_ok=True)


def plot_correlation(df):
    try:
        plt.figure(figsize=(10, 8))

        sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
        plt.title("Feature Correlation")

        plt.tight_layout()
        plt.savefig("models/correlation_matrix.png")
        plt.close()

        logger.info("Correlation plot saved")

    except Exception as e:
        logger.error(f"plot_correlation error: {e}")
        raise


def plot_loss_curve(model):
    try:
        plt.figure(figsize=(6, 4))

        plt.plot(model.loss_curve_)
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.title("Training Loss Curve")

        plt.tight_layout()
        plt.savefig("models/loss_curve.png")
        plt.close()

        logger.info("Loss curve saved")

    except Exception as e:
        logger.error(f"plot_loss_curve error: {e}")
        raise
