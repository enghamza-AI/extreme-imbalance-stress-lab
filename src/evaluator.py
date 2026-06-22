# src/evaluator.py
# PURPOSE: Take a trained model + test data, and measure the 4 things
# we care about: F1, Recall, Inference Time, Memory Usage.
# This file is the "judge" of the tournament — it doesn't care
# WHICH technique it's evaluating, just measures fairly and consistently.

import time
import tracemalloc   # built into Python — tracks memory allocations
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score


def measure_inference(model, X_test):
    
    tracemalloc.start()       
    start_time = time.perf_counter()   

    predictions = model.predict(X_test)

    elapsed = time.perf_counter() - start_time
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    peak_mb = peak / (1024 * 1024)   

    return predictions, elapsed, peak_mb


def compute_metrics(y_true, y_pred):

    return {
        "accuracy":  accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall":    recall_score(y_true, y_pred, zero_division=0),
        "f1":        f1_score(y_true, y_pred, zero_division=0),
    }


def evaluate_technique(technique_name, model, X_train, y_train, X_test, y_test):
   
    if X_train is None or y_train is None:
        return {
            "technique": technique_name,
            "status": "FAILED",
            "f1": None,
            "recall": None,
            "train_time_sec": None,
            "inference_time_sec": None,
            "peak_memory_mb": None,
        }

    tracemalloc.start()
    train_start = time.perf_counter()

    model.fit(X_train, y_train)

    train_time = time.perf_counter() - train_start
    _, train_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    
    predictions, inference_time, inference_peak_mb = measure_inference(model, X_test)

   
    metrics = compute_metrics(y_test, predictions)

    result = {
        "technique": technique_name,
        "status": "OK",
        "f1": round(metrics["f1"], 4),
        "recall": round(metrics["recall"], 4),
        "precision": round(metrics["precision"], 4),
        "accuracy": round(metrics["accuracy"], 4),
        "train_time_sec": round(train_time, 4),
        "inference_time_sec": round(inference_time, 4),
        "peak_memory_mb": round(inference_peak_mb, 2),
    }

    return result