
import time
import numpy as np
from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.linear_model import LogisticRegression


def apply_smote(X, y, random_state=42):
 
    minority_count = y.value_counts().min()
    if minority_count <= 5:
        k = max(1, minority_count - 1)
        print(f"WARNING: minority class has only {minority_count} samples. "
              f"Reducing SMOTE k_neighbors to {k}.")
        smote = SMOTE(random_state=random_state, k_neighbors=k)
    else:
        smote = SMOTE(random_state=random_state)

    X_res, y_res = smote.fit_resample(X, y)
    return X_res, y_res


def apply_adasyn(X, y, random_state=42):
 
    try:
        adasyn = ADASYN(random_state=random_state)
        X_res, y_res = adasyn.fit_resample(X, y)
        return X_res, y_res
    except RuntimeError as e:
       
        print(f"ADASYN failed: {e}. Skipping this technique for this run.")
        return None, None


def apply_ros(X, y, random_state=42):
 
    ros = RandomOverSampler(random_state=random_state)
    X_res, y_res = ros.fit_resample(X, y)
    return X_res, y_res


def apply_rus(X, y, random_state=42):
   
    rus = RandomUnderSampler(random_state=random_state)
    X_res, y_res = rus.fit_resample(X, y)
    return X_res, y_res


def get_class_weighted_model(y, random_state=42):
 
    model = LogisticRegression(
        class_weight='balanced',
        max_iter=1000,        
        random_state=random_state
    )
    return model


def get_balanced_rf_model(random_state=42):
  
    model = BalancedRandomForestClassifier(
        n_estimators=100,
        random_state=random_state,
        n_jobs=-1   
    )
    return model


TECHNIQUE_REGISTRY = {
    "SMOTE":        {"type": "resample", "func": apply_smote},
    "ADASYN":       {"type": "resample", "func": apply_adasyn},
    "ROS":          {"type": "resample", "func": apply_ros},
    "RUS":          {"type": "resample", "func": apply_rus},
    "class_weight": {"type": "model",    "func": get_class_weighted_model},
    "BalancedRF":   {"type": "model",    "func": get_balanced_rf_model},
}