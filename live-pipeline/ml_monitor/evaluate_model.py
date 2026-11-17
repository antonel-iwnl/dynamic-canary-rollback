"""
Offline & Post-Run Evaluation and Visualization Script for Live Canary Telemetry.
Reads `collected_metrics.csv`, runs the preprocessing + IsolationForest pipeline (when sklearn/pandas
are available, or evaluates recorded predictions directly), computes validation metrics
(Precision, Recall, F1, False Positive Rate, Confusion Matrix), and generates a 4-panel
time-series validation figure (`ml_monitor/live_validation_results.png`).
"""

import csv
import os
import sys

import numpy as np


def evaluate_and_plot(
    csv_path: str = "collected_metrics.csv",
    plot_path: str = "ml_monitor/live_validation_results.png",
    contamination: float = 0.02,
) -> None:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Telemetry file not found: {csv_path}")

    # If pandas and scikit-learn are available in this interpreter, re-run full pipeline
    try:
        import pandas as pd

        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from monitor import evaluate_model, preprocess_data, train_baseline

        df = pd.read_csv(csv_path)
        stable_df = df[df["version"] == "stable"]
        model = train_baseline(
            stable_df,
            n_samples=len(stable_df),
            contamination=contamination,
        )
        clean_all = preprocess_data(df)
        df["prediction"] = model.predict(clean_all)
        df["score"] = model.decision_function(clean_all)
        df.to_csv(csv_path, index=False)
        evaluate_model(df)
    except ImportError:
        pass

    # Load CSV via standard library for universal plotting & metrics reporting
    versions, cpu, mem, lat, preds, scores = [], [], [], [], [], []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            versions.append(row["version"])
            cpu.append(float(row["cpu_usage"]))
            mem.append(float(row["memory_usage"]))
            lat.append(float(row["request_latency"]))
            preds.append(int(float(row["prediction"])))
            scores.append(float(row["score"]))

    versions_arr = np.array(versions)
    cpu_arr = np.array(cpu)
    mem_arr = np.array(mem)
    lat_arr = np.array(lat)
    preds_arr = np.array(preds)
    scores_arr = np.array(scores)

    y_true = np.where(versions_arr == "canary", 1, 0)
    y_pred = np.where(preds_arr == -1, 1, 0)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    stable_mask = versions_arr == "stable"
    canary_mask = versions_arr == "canary"

    print("=================================================================")
    print(" Live Telemetry Dataset Summary (Stable vs. Canary)")
    print("=================================================================")
    for name, arr in [
        ("CPU Usage (%)", cpu_arr),
        ("Memory Usage (MB)", mem_arr),
        ("Request Latency (s)", lat_arr),
    ]:
        print(
            f" {name:20s} | Stable: {arr[stable_mask].mean():.4f} ± {arr[stable_mask].std():.4f} "
            f"| Canary: {arr[canary_mask].mean():.4f} ± {arr[canary_mask].std():.4f}"
        )
    print("-----------------------------------------------------------------")
    print(f" Total Samples : {len(versions_arr)} (Stable={tn + fp}, Canary={tp + fn})")
    print(f" Confusion Mat : TP={tp}, FP={fp}, TN={tn}, FN={fn}")
    print(f" Precision     : {precision:.4f}")
    print(f" Recall        : {recall:.4f}")
    print(f" F1-Score      : {f1:.4f}")
    print(f" False Pos Rate: {fpr:.4f} ({fpr * 100:.2f}%)")
    print("=================================================================")

    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
        steps = np.arange(len(versions_arr))
        canary_indices = np.where(canary_mask)[0]
        canary_start = int(canary_indices[0]) if len(canary_indices) > 0 else len(versions_arr)
        anomaly_idx = np.where(preds_arr == -1)[0]

        panels = [
            (cpu_arr, "CPU Usage (%)", "#1f77b4"),
            (mem_arr, "Memory Usage (MB)", "#2ca02c"),
            (lat_arr, "Request Latency (s)", "#ff7f0e"),
            (scores_arr, "IsolationForest Score s(x)", "#9467bd"),
        ]

        for ax, (arr, ylabel, color) in zip(axes, panels):
            ax.plot(steps, arr, color=color, linewidth=1.5, alpha=0.85, label=ylabel)
            ax.scatter(
                anomaly_idx,
                arr[anomaly_idx],
                color="crimson",
                s=22,
                zorder=5,
                label="Predicted Anomaly (-1)",
            )
            if canary_start < len(versions_arr):
                ax.axvline(
                    canary_start,
                    color="black",
                    linestyle="--",
                    linewidth=1.3,
                    label="Canary Deployment Start (Step 500)",
                )
            if ylabel.startswith("IsolationForest"):
                ax.axhline(
                    0.0,
                    color="crimson",
                    linestyle=":",
                    linewidth=1.2,
                    label="Decision Boundary (score = 0)",
                )
            ax.set_ylabel(ylabel, fontsize=10, fontweight="bold")
            ax.grid(True, linestyle="--", alpha=0.35)
            ax.legend(loc="upper left", fontsize=9)

        axes[0].set_title(
            "Live CI/CD Canary Pipeline Validation — Telemetry & IsolationForest Anomaly Triggers",
            fontsize=13,
            fontweight="bold",
        )
        axes[-1].set_xlabel(
            "Telemetry Polling Step (2-second intervals)",
            fontsize=11,
            fontweight="bold",
        )

        plt.tight_layout()
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)
        plt.savefig(plot_path, dpi=200)
        plt.close(fig)
        print(f"[PLOT] Saved 4-panel validation figure to {plot_path}")
    except ImportError:
        print("[WARN] matplotlib not installed in current interpreter; skipping plot generation.")


if __name__ == "__main__":
    evaluate_and_plot()
