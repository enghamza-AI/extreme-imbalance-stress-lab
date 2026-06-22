
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd


def analyze_dataset_stats(X, y):
 
    n_rows = X.shape[0]
    n_features = X.shape[1]

    class_counts = y.value_counts()
    minority_count = class_counts.min()
    majority_count = class_counts.max()

    
    imbalance_ratio = majority_count / minority_count if minority_count > 0 else float('inf')

    return {
        "n_rows": n_rows,
        "n_features": n_features,
        "imbalance_ratio": imbalance_ratio,
        "minority_count": minority_count,
    }


def recommend_technique(stats, results_df):

    ratio = stats["imbalance_ratio"]
    n_rows = stats["n_rows"]
    minority_count = stats["minority_count"]

   
    valid_results = results_df[results_df["status"] == "OK"]
    if valid_results.empty:
        return "class_weight", "No valid tournament results found — defaulting to safest option (no resampling needed)."

    best_f1_row = valid_results.sort_values("f1", ascending=False).iloc[0]
    best_technique_overall = best_f1_row["technique"]

    
    if minority_count <= 5:
        return "class_weight", (
            f"Minority class has only {minority_count} samples — too few "
            f"for SMOTE/ADASYN to find reliable neighbors. class_weight "
            f"needs no synthetic generation, so it's the safe default."
        )

   
    if n_rows > 100_000 and ratio > 50:
        rus_row = valid_results[valid_results["technique"] == "RUS"]
        if not rus_row.empty and rus_row.iloc[0]["recall"] >= 0.7:
            return "RUS", (
                f"Dataset is large ({n_rows:,} rows) with extreme imbalance "
                f"({ratio:.1f}:1). RUS keeps recall acceptable while cutting "
                f"training time and memory significantly — good tradeoff at scale."
            )

   
    if ratio <= 20:
        smote_row = valid_results[valid_results["technique"] == "SMOTE"]
        if not smote_row.empty:
            return "SMOTE", (
                f"Imbalance ratio is moderate ({ratio:.1f}:1). SMOTE generates "
                f"realistic synthetic diversity without needing to discard data."
            )

    
    return best_technique_overall, (
        f"No specific rule matched cleanly — falling back to the empirically "
        f"best performer from your tournament: {best_technique_overall} "
        f"(F1={best_f1_row['f1']})."
    )


def plot_flowchart(stats, recommendation, reason, save_path="outputs/flowchart.png"):
    """
    Draws a simple visual flowchart showing the decision path taken.
    Built with matplotlib shapes — no extra library needed.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    def draw_box(x, y, text, color="#dbe9ff"):
        box = patches.FancyBboxPatch(
            (x, y), 3.2, 1.2,
            boxstyle="round,pad=0.1",
            edgecolor="black", facecolor=color
        )
        ax.add_patch(box)
        ax.text(x + 1.6, y + 0.6, text, ha="center", va="center", fontsize=9, wrap=True)

    draw_box(0.3, 8, f"Dataset stats:\n{stats['n_rows']:,} rows\nratio {stats['imbalance_ratio']:.1f}:1", "#fff3cd")
    draw_box(3.9, 8, f"Minority count:\n{stats['minority_count']}", "#fff3cd")
    draw_box(2.1, 5, "Decision Logic\n(rules engine)", "#d1ecf1")
    draw_box(2.1, 2, f"Recommended:\n{recommendation}", "#d4edda")

    ax.annotate("", xy=(3.7, 6.2), xytext=(2.0, 8.0), arrowprops=dict(arrowstyle="->"))
    ax.annotate("", xy=(3.7, 6.2), xytext=(5.5, 8.0), arrowprops=dict(arrowstyle="->"))
    ax.annotate("", xy=(3.7, 3.2), xytext=(3.7, 5.0), arrowprops=dict(arrowstyle="->"))

    ax.text(0.3, 0.5, f"Reason: {reason}", fontsize=8, wrap=True)

    plt.title("Technique Selection Flowchart", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Flowchart saved to {save_path}")
    plt.close()


if __name__ == "__main__":
    from src.data_loader import get_clean_data

    X, y = get_clean_data()
    stats = analyze_dataset_stats(X, y)

    results_df = pd.read_csv("outputs/tournament_results.csv")
    technique, reason = recommend_technique(stats, results_df)

    print(f"\nRecommended technique: {technique}")
    print(f"Reason: {reason}")

    plot_flowchart(stats, technique, reason)