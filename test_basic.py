import pandas as pd
from src.techniques import apply_smote, apply_ros, apply_rus, TECHNIQUE_REGISTRY
from src.evaluator import compute_metrics


def make_fake_imbalanced_data():
   
    X = pd.DataFrame({
        "feature1": list(range(100)),
        "feature2": list(range(100, 200)),
    })
  
    y = pd.Series([0] * 95 + [1] * 5)
    return X, y


def test_smote_balances_classes():
    X, y = make_fake_imbalanced_data()
    X_res, y_res = apply_smote(X, y)
    counts = y_res.value_counts()
    
    assert counts[0] == counts[1], "SMOTE did not balance the classes"


def test_ros_balances_classes():
    X, y = make_fake_imbalanced_data()
    X_res, y_res = apply_ros(X, y)
    counts = y_res.value_counts()
    assert counts[0] == counts[1], "ROS did not balance the classes"


def test_rus_balances_and_shrinks():
    X, y = make_fake_imbalanced_data()
    X_res, y_res = apply_rus(X, y)
    counts = y_res.value_counts()
    assert counts[0] == counts[1], "RUS did not balance the classes"
    # RUS should result in FEWER total rows than original
    assert len(y_res) < len(y), "RUS should reduce total row count"


def test_compute_metrics_handles_perfect_predictions():
    y_true = [0, 0, 1, 1]
    y_pred = [0, 0, 1, 1]
    metrics = compute_metrics(y_true, y_pred)
    assert metrics["f1"] == 1.0
    assert metrics["accuracy"] == 1.0


def test_compute_metrics_handles_all_wrong_predictions():
   
    y_true = [0, 0, 0, 0]
    y_pred = [1, 1, 1, 1]
    metrics = compute_metrics(y_true, y_pred)
    assert metrics["precision"] == 0.0
    assert metrics["f1"] == 0.0


def test_technique_registry_has_all_6():
    assert len(TECHNIQUE_REGISTRY) == 6
    expected = {"SMOTE", "ADASYN", "ROS", "RUS", "class_weight", "BalancedRF"}
    assert set(TECHNIQUE_REGISTRY.keys()) == expected


if __name__ == "__main__":
   
    print("Run this file with: python -m pytest tests/test_basic.py -v")