import numpy as np
from sklearn.metrics import precision_score, recall_score
import time


def compute_far_frr(y_true, y_pred):
    """
    Function computes FAR and FRR metrics:
        FAR = False Acceptance Rate = FP / (FP + TN)
        FRR = False Rejection Rate = FN / (FN + TP)

    Args:
        y_true (array): True labels
        y_pred (array): Predicted labels
    Returns:
        float: FAR, FRR
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))

    far = FP / (FP + TN) if (FP + TN) > 0 else 0.0
    frr = FN / (FN + TP) if (FN + TP) > 0 else 0.0

    return far, frr


def evaluate_model(model, X_test, y_test):
    """
    Function evaluates model performance with FAR, FRR, precision and recall metrics.
    Args:
        model_path (str): Path to saved model.
        X_test (array): Test data.
        y_test (array): Test labels.
    Returns:
        float: FAR, FRR, precision, recall
    """

    y_pred = model.predict(X_test)

    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    far, frr = compute_far_frr(y_test, y_pred)

    return far, frr, precision, recall


def measure_prediction_time(model, X_test, n_runs=3):
    """
    Function measure model prediction time
    Args:
         model_path (str): Path to saved model.
         X_test (array): Test data.
         n_runs (int): Number of runs to average the results.
    Returns:
        float: Average model prediction time.
    """

    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        model.predict(X_test)
        end = time.perf_counter()
        times.append(end - start)

    avg_time = sum(times) / n_runs
    return avg_time
