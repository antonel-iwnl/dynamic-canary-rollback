import argparse
import os
import re
import time
from datetime import datetime
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURE_COLUMNS = ["cpu_usage", "memory_usage", "request_latency"]

PROMETHEUS_METRIC_MAP = {
    "app_cpu_usage_percent": "cpu_usage",
    "app_memory_usage_mb": "memory_usage",
    "app_request_latency_seconds": "request_latency",
}


def fetch_metrics(metrics_url: str, timeout: float = 5.0) -> Optional[Dict[str, float]]:
    """
    Fetches Prometheus text-format metrics from the given `/metrics` endpoint
    and extracts CPU usage, memory usage, and request latency.
    """
    try:
        response = requests.get(metrics_url, timeout=timeout)
        response.raise_for_status()
    except requests.RequestException as exc:
        print(f"[WARN] Could not reach {metrics_url}: {exc}")
        return None

    extracted: Dict[str, float] = {}
    for line in response.text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        for prom_name, col_name in PROMETHEUS_METRIC_MAP.items():
            if line.startswith(prom_name):
                parts = re.split(r"\s+", line)
                if len(parts) >= 2:
                    try:
                        extracted[col_name] = float(parts[-1])
                    except ValueError:
                        pass

    if all(col in extracted for col in FEATURE_COLUMNS):
        return extracted

    print(f"[WARN] Incomplete metrics payload from {metrics_url}: {extracted}")
    return None


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Data Preprocessing Step:
    1. Selects the relevant system telemetry features (CPU %, Memory MB, Request Latency s).
    2. Replaces any infinite values with NaN and imputes missing values via forward-fill
       followed by median fallback.
    3. Clips physically impossible negative readings to 0.0.
    """
    clean_df = df[FEATURE_COLUMNS].copy()
    clean_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    clean_df.ffill(inplace=True)
    clean_df.fillna(clean_df.median(numeric_only=True), inplace=True)
    clean_df = clean_df.clip(lower=0.0)
    return clean_df


def train_baseline(
    df: pd.DataFrame,
    n_samples: Optional[int] = None,
    contamination: float = 0.02,
    random_state: int = 42,
) -> Pipeline:
    """
    Model Training Step:
    1. Preprocesses the collected stable baseline data points.
    2. Standardizes features using `StandardScaler` (zero mean, unit variance) so metrics
       with different units (% vs. MB vs. seconds) are normalized cleanly.
    3. Trains a scikit-learn `IsolationForest` on the preprocessed baseline data.
    """
    raw_baseline = df.head(n_samples) if n_samples else df
    baseline_features = preprocess_data(raw_baseline)

    print(
        f"\n[BASELINE] Preprocessing & training IsolationForest on {len(baseline_features)} stable data points..."
    )
    print(
        f"[BASELINE] Feature summary:\n{baseline_features.describe().loc[['mean', 'std', 'min', 'max']]}\n"
    )

    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "isolation_forest",
                IsolationForest(
                    n_estimators=100,
                    max_samples="auto",
                    contamination=contamination,
                    random_state=random_state,
                ),
            ),
        ]
    )
    model.fit(baseline_features)
    print("[BASELINE] IsolationForest pipeline training complete.\n")
    return model


def monitor_canary(
    model: Pipeline,
    data_point: pd.DataFrame,
) -> Tuple[int, float]:
    """
    Evaluates new incoming canary data point(s) using the trained preprocessing + IsolationForest pipeline.
    Runs `model.predict()` and prints the rollback alert if an anomaly (-1) is detected.

    Returns:
        (prediction, anomaly_score) where prediction is 1 (normal) or -1 (anomaly).
    """
    features = preprocess_data(data_point)
    prediction = int(model.predict(features)[0])
    score = float(model.decision_function(features)[0])

    row = features.iloc[0]
    metrics_str = (
        f"CPU={row['cpu_usage']:.2f}%, "
        f"MEM={row['memory_usage']:.2f}MB, "
        f"LATENCY={row['request_latency']:.4f}s "
        f"(score={score:.4f})"
    )

    if prediction == -1:
        print(f"[ALERT] {metrics_str} -> ANOMALY DETECTED: INITIATING ROLLBACK")
    else:
        print(f"[OK]    {metrics_str} -> Canary operating within normal baseline.")

    return prediction, score


def evaluate_model(metrics_df: pd.DataFrame) -> None:
    """
    Model Evaluation & Validation Step:
    Compares IsolationForest predictions against ground-truth deployment labels
    (`stable` = 0 normal, `canary` = 1 anomalous) and prints Precision, Recall,
    F1-Score, False Positive Rate, and the Confusion Matrix.
    """
    if "version" not in metrics_df.columns or len(metrics_df) == 0:
        return

    y_true = np.where(metrics_df["version"] == "canary", 1, 0)
    y_pred = np.where(metrics_df["prediction"] == -1, 1, 0)

    if len(np.unique(y_true)) < 2:
        print("[EVALUATION] Only one class present in telemetry; skipping binary classification metrics.")
        return

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    print("\n=================================================================")
    print(" Phase 3: Model Evaluation & Live Validation Summary")
    print("=================================================================")
    print(f" Total Telemetry Samples : {len(metrics_df)} (Stable={tn + fp}, Canary={tp + fn})")
    print(f" True Positives  (TP)    : {tp:4d} | False Positives (FP) : {fp:4d}")
    print(f" True Negatives  (TN)    : {tn:4d} | False Negatives (FN) : {fn:4d}")
    print(f" Precision               : {precision:.4f}")
    print(f" Recall (Sensitivity)    : {recall:.4f}")
    print(f" F1-Score                : {f1:.4f}")
    print(f" False Positive Rate     : {fpr:.4f} ({fpr * 100:.2f}%)")
    print("=================================================================\n")


def run_pipeline(
    stable_url: str,
    canary_url: str,
    poll_interval: float = 2.0,
    baseline_samples: int = 100,
    contamination: float = 0.02,
    max_canary_samples: Optional[int] = None,
    rollback_threshold: int = 3,
    csv_output: str = "collected_metrics.csv",
) -> None:
    """
    Full monitoring pipeline:
    1. Polls `stable_url` every `poll_interval` seconds until `baseline_samples`
       data points are accumulated in a Pandas DataFrame.
    2. Calls `train_baseline()` to preprocess and fit the IsolationForest model,
       then records baseline predictions and decision scores.
    3. Polls `canary_url` every `poll_interval` seconds, accumulates points into the DataFrame,
       and passes each new point to `monitor_canary()`.
    4. Runs `evaluate_model()` to report live validation metrics and exports the CSV.
    """
    metrics_df = pd.DataFrame(
        columns=["timestamp", "version", *FEATURE_COLUMNS, "prediction", "score"]
    )

    print("=================================================================")
    print(" Phase 1: Collecting Baseline Telemetry from Stable API")
    print(f" Target URL   : {stable_url}")
    print(f" Poll Interval: {poll_interval}s | Target Samples: {baseline_samples}")
    print("=================================================================")

    while len(metrics_df) < baseline_samples:
        sample = fetch_metrics(stable_url)
        if sample is not None:
            record = {
                "timestamp": datetime.utcnow().isoformat(),
                "version": "stable",
                **sample,
                "prediction": 1,
                "score": 0.0,
            }
            metrics_df.loc[len(metrics_df)] = record
            idx = len(metrics_df)
            print(
                f"[COLLECT {idx:03d}/{baseline_samples:03d}] "
                f"CPU={sample['cpu_usage']:.2f}% | "
                f"MEM={sample['memory_usage']:.2f}MB | "
                f"LATENCY={sample['request_latency']:.4f}s"
            )
        time.sleep(poll_interval)

    # Train IsolationForest on the collected baseline data points
    model = train_baseline(
        metrics_df,
        n_samples=baseline_samples,
        contamination=contamination,
    )

    # Backfill actual model predictions & anomaly scores onto the baseline rows
    clean_baseline = preprocess_data(metrics_df)
    metrics_df["prediction"] = model.predict(clean_baseline)
    metrics_df["score"] = model.decision_function(clean_baseline)

    print("=================================================================")
    print(" Phase 2: Monitoring Canary Deployment")
    print(f" Target URL   : {canary_url}")
    print(
        f" Poll Interval: {poll_interval}s | Consecutive Alert Rollback: {rollback_threshold}"
    )
    print("=================================================================")

    consecutive_anomalies = 0
    canary_count = 0

    try:
        while max_canary_samples is None or canary_count < max_canary_samples:
            sample = fetch_metrics(canary_url)
            if sample is not None:
                canary_count += 1
                single_point_df = pd.DataFrame([sample])
                pred, score = monitor_canary(model, single_point_df)

                record = {
                    "timestamp": datetime.utcnow().isoformat(),
                    "version": "canary",
                    **sample,
                    "prediction": pred,
                    "score": score,
                }
                metrics_df.loc[len(metrics_df)] = record

                if pred == -1:
                    consecutive_anomalies += 1
                    if consecutive_anomalies >= rollback_threshold:
                        print(
                            f"\n[ROLLBACK TRIGGERED] {consecutive_anomalies} consecutive anomalies "
                            f"detected on canary ({canary_url}). Rolling back traffic to stable ({stable_url})!"
                        )
                        if max_canary_samples is None:
                            break
                else:
                    consecutive_anomalies = 0

            time.sleep(poll_interval)
    except KeyboardInterrupt:
        print("\n[INFO] Monitoring stopped by user.")
    finally:
        evaluate_model(metrics_df)
        if csv_output:
            metrics_df.to_csv(csv_output, index=False)
            print(f"[INFO] Saved {len(metrics_df)} telemetry records to {csv_output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Dynamic Canary Rollback ML Monitor using IsolationForest"
    )
    parser.add_argument(
        "--stable-url",
        default=os.getenv("STABLE_METRICS_URL", "http://localhost:8000/metrics"),
        help="Prometheus /metrics URL for the stable API",
    )
    parser.add_argument(
        "--canary-url",
        default=os.getenv("CANARY_METRICS_URL", "http://localhost:8001/metrics"),
        help="Prometheus /metrics URL for the canary API",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=float(os.getenv("POLL_INTERVAL", "2.0")),
        help="Polling interval in seconds (default: 2.0)",
    )
    parser.add_argument(
        "--baseline-samples",
        type=int,
        default=int(os.getenv("BASELINE_SAMPLES", "100")),
        help="Number of stable baseline samples to collect before training (default: 100)",
    )
    parser.add_argument(
        "--contamination",
        type=float,
        default=float(os.getenv("CONTAMINATION", "0.02")),
        help="IsolationForest contamination hyperparameter (default: 0.02)",
    )
    parser.add_argument(
        "--rollback-threshold",
        type=int,
        default=int(os.getenv("ROLLBACK_THRESHOLD", "3")),
        help="Consecutive anomalies required to halt canary and finalize rollback (default: 3)",
    )
    parser.add_argument(
        "--max-canary-samples",
        type=int,
        default=None,
        help="Optional cap on canary samples to evaluate before exiting",
    )
    parser.add_argument(
        "--csv-output",
        default=os.getenv("CSV_OUTPUT", "collected_metrics.csv"),
        help="Path to save collected telemetry DataFrame as CSV",
    )
    args = parser.parse_args()

    run_pipeline(
        stable_url=args.stable_url,
        canary_url=args.canary_url,
        poll_interval=args.interval,
        baseline_samples=args.baseline_samples,
        contamination=args.contamination,
        max_canary_samples=args.max_canary_samples,
        rollback_threshold=args.rollback_threshold,
        csv_output=args.csv_output,
    )
