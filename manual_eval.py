"""
Stretch Tuesday — Manual Evaluation Harness.

Implement these without using Trainer.predict, sklearn metrics helpers, or
Hugging Face evaluate. The goal is to make the math explicit.
"""

import numpy as np
import torch
import os
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from calibration import reliability_diagram, expected_calibration_error, plot_reliability

def manual_predict(model, tokenizer, texts: list, batch_size: int = 8):
    """
    Run manual PyTorch inference over a list of texts.

    Returns (preds, probs):
      preds: shape (N,), int class indices
      probs: shape (N, num_classes), probabilities (post-softmax)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    all_preds = []
    all_probs = []

    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            
            inputs = tokenizer(
                batch_texts, 
                truncation=True, 
                max_length=128, 
                padding=True, 
                return_tensors='pt'
            ).to(device)

            outputs = model(**inputs)
            logits = outputs.logits
            
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)

            all_preds.append(preds.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    return np.concatenate(all_preds), np.concatenate(all_probs)


def compute_classification_report_from_arrays(y_true, y_pred) -> dict:
    """
    Compute accuracy, per-class precision/recall/F1, and macro-F1 from numpy
    primitives only — no sklearn, no Hugging Face evaluate.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    classes = np.unique(y_true)
    
    accuracy = np.mean(y_pred == y_true)
    per_class = {}
    f1_scores = []

    for cls in classes:
        tp = np.sum((y_pred == cls) & (y_true == cls))
        fp = np.sum((y_pred == cls) & (y_true != cls))
        fn = np.sum((y_pred != cls) & (y_true == cls))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        per_class[int(cls)] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1)
        }
        f1_scores.append(f1)

    macro_f1 = np.mean(f1_scores) if f1_scores else 0.0

    return {
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "per_class": per_class,
    }

if __name__ == "__main__":

    model_path = "./model" 
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)

    # Use more samples for better calibration visualization if possible
    test_texts = [
        "I love this project!", 
        "This is very difficult and confusing.",
        "It is okay, but could be better.",
        "Absolutely fantastic experience.",
        "I am not sure how I feel about this."
    ]
    test_labels = [1, 0, 2, 1, 2] 

    print("--- Running Manual Predict ---")
    preds, probs = manual_predict(model, tokenizer, test_texts)
    print(f"Predictions: {preds}")
    print(f"Probabilities:\n{probs}")

    print("\n--- Running Metrics Report ---")
    report = compute_classification_report_from_arrays(test_labels, preds)
    print(report)

    print("\n--- Running Calibration Analysis ---")
    ece_value = expected_calibration_error(probs, np.array(test_labels))
    print(f"Expected Calibration Error (ECE): {ece_value:.4f}")

    centers, accs, counts = reliability_diagram(probs, np.array(test_labels))
    
    if not os.path.exists("figures"):
        os.makedirs("figures")
    
    output_plot = "figures/reliability-diagram.png"
    plot_reliability(centers, accs, counts, output_plot)
    print(f"Reliability diagram saved to: {output_plot}")