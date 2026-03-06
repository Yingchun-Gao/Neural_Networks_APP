import matplotlib.pyplot as plt
import seaborn as sns
import os

# ensure models folder exists
os.makedirs("models", exist_ok=True)


def plot_correlation(df):

    plt.figure(figsize=(10, 8))

    sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
    plt.title("Feature Correlation")

    plt.tight_layout()
    plt.savefig("models/correlation_matrix.png")

    plt.close()


def plot_loss_curve(model):

    plt.figure(figsize=(6, 4))

    plt.plot(model.loss_curve_)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("models/loss_curve.png")

    plt.close()
