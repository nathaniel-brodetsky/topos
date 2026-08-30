import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_decision_distribution(decision_counts: dict, output_path: str = "reporting/decision_distribution.png"):
    labels = list(decision_counts.keys())
    values = list(decision_counts.values())

    fig, ax = plt.subplots()
    ax.bar(labels, values)
    ax.set_ylabel("tick count")
    ax.set_title("Decision distribution, live window")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path