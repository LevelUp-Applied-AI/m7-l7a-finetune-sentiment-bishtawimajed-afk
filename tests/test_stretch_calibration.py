import numpy as np
import pytest
import torch
from manual_eval import compute_classification_report_from_arrays
from calibration import expected_calibration_error, reliability_diagram

def test_metrics_accuracy_and_f1():
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 2, 2, 1, 0])
    
    report = compute_classification_report_from_arrays(y_true, y_pred)
    
    # Corrected: 4 out of 6 matches found
    expected_acc = 4 / 6 
    assert report["accuracy"] == pytest.approx(expected_acc)
    assert "macro_f1" in report
    assert isinstance(report["per_class"], dict)

def test_ece_perfect_calibration():
    # Scenario: Predictions perfectly match labels with 100% confidence
    probs = np.array([[1.0, 0.0], [0.0, 1.0]])
    y_true = np.array([0, 1])
    
    ece = expected_calibration_error(probs, y_true, n_bins=2)
    assert ece == pytest.approx(0.0)

def test_ece_high_error():
    # Scenario: High confidence (1.0) but all predictions are wrong
    probs = np.array([[1.0, 0.0], [1.0, 0.0]])
    y_true = np.array([1, 1])
    
    ece = expected_calibration_error(probs, y_true, n_bins=1)
    # Confidence is 1.0, Accuracy is 0.0 -> |0 - 1| * 1.0 = 1.0
    assert ece == pytest.approx(1.0)

def test_reliability_diagram_output_format():
    n_bins = 10
    probs = np.random.dirichlet(np.ones(3), size=20)
    y_true = np.random.randint(0, 3, size=20)
    
    centers, accs, counts = reliability_diagram(probs, y_true, n_bins=n_bins)
    
    assert isinstance(centers, np.ndarray)
    assert isinstance(accs, np.ndarray)
    assert isinstance(counts, np.ndarray)
    assert len(centers) == n_bins
    assert len(accs) == n_bins
    assert len(counts) == n_bins
    assert np.sum(counts) == 20

def test_metrics_zero_division_guard():
    # Scenario: Model never predicts class 1 (Precision undefined)
    y_true = [0, 0, 1]
    y_pred = [0, 0, 0]
    
    report = compute_classification_report_from_arrays(y_true, y_pred)
    # Precision for class 1 should be 0.0 instead of crashing
    assert report["per_class"][1]["precision"] == 0.0