"""
Stretch Tuesday — Calibration Analysis.

Reliability diagram + Expected Calibration Error (ECE).
"""

import numpy as np


def reliability_diagram(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 10):
    """
    Bin predictions by max predicted probability; compute empirical accuracy per bin.

    Returns (bucket_centers, bucket_accuracies, bucket_counts), all length n_bins.
    """
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bucket_centers = (bin_boundaries[:-1] + bin_boundaries[1:]) / 2
    
    bucket_accuracies = np.zeros(n_bins)
    bucket_counts = np.zeros(n_bins)

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        if i == n_bins - 1:
            in_bin = (confidences >= bin_lower) & (confidences <= bin_upper)
        else:
            in_bin = (confidences >= bin_lower) & (confidences < bin_upper)
            
        bin_count = np.sum(in_bin)
        bucket_counts[i] = bin_count
        
        if bin_count > 0:
            bucket_accuracies[i] = np.mean(predictions[in_bin] == y_true[in_bin])
        else:
            bucket_accuracies[i] = 0.0

    return bucket_centers, bucket_accuracies, bucket_counts


def expected_calibration_error(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 10) -> float:
    """
    ECE = sum over bins of (bucket_count / N) * |bucket_accuracy - bucket_confidence|.

    A perfectly calibrated model has ECE = 0.
    """
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    
    ece = 0.0
    n_samples = len(y_true)
    
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        if i == n_bins - 1:
            in_bin = (confidences >= bin_lower) & (confidences <= bin_upper)
        else:
            in_bin = (confidences >= bin_lower) & (confidences < bin_upper)
            
        bin_count = np.sum(in_bin)
        
        if bin_count > 0:
            accuracy_in_bin = np.mean(predictions[in_bin] == y_true[in_bin])
            confidence_in_bin = np.mean(confidences[in_bin])
            ece += (bin_count / n_samples) * np.abs(accuracy_in_bin - confidence_in_bin)
            
    return float(ece)


def plot_reliability(centers: np.ndarray, accs: np.ndarray, counts: np.ndarray, output_path: str) -> None:
    """Save a reliability diagram. Provided helper — do not modify."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 5))
    width = 1.0 / max(len(centers), 1)
    ax.bar(centers, accs, width=width * 0.9, edgecolor="black", alpha=0.8, label="Empirical accuracy")
    ax.plot([0, 1], [0, 1], "--", color="grey", label="Perfect calibration")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Predicted probability (bucket center)")
    ax.set_ylabel("Empirical accuracy")
    ax.set_title("Reliability diagram")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)