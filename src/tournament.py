

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from src.data_loader import get_clean_data
from src.techniques import TECHNIQUE_REGISTRY
from src.evaluator import evaluate_technique


def run_tournament(test_size=0.2, random_state=42):
 
    print("Loading data for tournament...\n")
    X, y = get_clean_data()

 
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    print(f"Train set: {X_train.shape[0]} rows")
    print(f"Test set:  {X_test.shape[0]} rows")
    print(f"Train imbalance ratio: {(y_train==0).sum() / max((y_train==1).sum(),1):.1f} : 1\n")

    results = []

    for name, config in TECHNIQUE_REGISTRY.items():
        print(f"Running technique: {name} ...")

        technique_type = config["type"]
        func = config["func"]

        if technique_type == "resample":
            \
            X_train_res, y_train_res = func(X_train, y_train, random_state=random_state)

            
            model = LogisticRegression(max_iter=1000, random_state=random_state)

            result = evaluate_technique(
                name, model, X_train_res, y_train_res, X_test, y_test
            )

        elif technique_type == "model":
            
            if name == "class_weight":
                model = func(y_train, random_state=random_state)
            else:
                model = func(random_state=random_state)

            result = evaluate_technique(
                name, model, X_train, y_train, X_test, y_test
            )

        else:
           
            raise ValueError(f"Unknown technique type '{technique_type}' for {name}")

        results.append(result)
        print(f"  -> F1: {result['f1']}, Recall: {result['recall']}, "
              f"Status: {result['status']}\n")

    results_df = pd.DataFrame(results)

    
    results_df = results_df.sort_values(by="f1", ascending=False, na_position="last")
    results_df = results_df.reset_index(drop=True)

    return results_df


if __name__ == "__main__":
    results = run_tournament()
    print("\n========== FINAL TOURNAMENT RESULTS ==========")
    print(results.to_string(index=False))

    # Save results so flowchart.py and the streamlit app can use them later
    results.to_csv("outputs/tournament_results.csv", index=False)
    print("\nResults saved to outputs/tournament_results.csv")