# Project 2 Report: Automated CI/CD Canary Rollback Using Unsupervised Machine Learning (`IsolationForest`)

---

## 1. Overview of the Software Engineering Problem Addressed

### 1.1 Background: Continuous Deployment and Canary Releases
In modern software engineering, Continuous Integration and Continuous Deployment (CI/CD) pipelines enable engineering teams to deliver code changes to production rapidly and frequently. However, deploying new software versions directly to 100% of production servers introduces significant operational risk: a single undetected performance bug, memory leak, or algorithmic regression can cause system-wide outages.

To mitigate this risk, organizations use **Canary Deployments**. In a canary deployment, a newly released version of a microservice (the **canary**) is deployed side-by-side with the existing proven version (the **stable** baseline). A small fraction of live user traffic (typically 10%–20%) is routed to the canary while 80%–90% continues to hit the stable release. Site Reliability Engineers (SREs) and automated deployment controllers monitor the canary's runtime telemetry—such as CPU utilization, memory consumption, request throughput, and response latency—to decide whether to promote the canary to 100% of production or abort and **roll back** to the stable version.

### 1.2 The Challenge with Traditional Static Thresholds
Historically, canary health checks have relied on **manual inspection** or **static, single-metric threshold rules** (for example, *"Alert if CPU > 65%"* or *"Rollback if Latency > 500ms"*). Static rules fail in real-world production environments for two primary reasons:
1. **The Traffic Confounding Problem (False Rollbacks During User Surges)**: Server workloads are non-stationary. When a legitimate influx of users (a "flash crowd") arrives at peak hours, CPU utilization naturally climbs (e.g., from `10%` to `70%`). A static CPU rule cannot distinguish between a healthy server working harder to serve 120 requests/second and a broken server spinning at 70% CPU under only 15 requests/second, resulting in false alarms and unnecessary rollbacks.
2. **Blindness to Multi-Dimensional Regressions**: Many software regressions manifest across multiple metrics simultaneously (for example, a memory leak accompanied by elevated tail latency and disproportionate CPU cost per request). Evaluating metrics in isolation misses subtle joint anomalies until the container crashes.

### 1.3 Proposed Machine Learning Solution
This project develops and validates an **Automated, Traffic-Aware CI/CD Canary Rollback System** powered by the unsupervised **Isolation Forest (`IsolationForest`)** algorithm. By learning the joint multivariate baseline distribution of healthy production telemetry across fluctuating traffic waves, the system automatically distinguishes benign user traffic surges from genuine canary software regressions and initiates an immediate rollback when anomalous degradation is detected.

---

## 2. Data Collection Methodology

To ensure both **generalizability to real-world production servers** and **end-to-end validation in a live microservice pipeline**, a two-phase data collection strategy was designed and executed.

### 2.1 Phase A: Offline Real-World Server Benchmark (Server Machine Dataset — SMD)
Before building the live containerized environment, the `IsolationForest` approach was evaluated on a publicly available, industry-standard dataset: the **Server Machine Dataset (SMD)** (`machine-1-1.txt`).
- **Dataset Origin & Scale**: Collected from real production server clusters and open-sourced for multivariate time-series anomaly detection research. The subset used (`machine-1-1`) contains **28,479 training observations** (representing normal baseline server operation) and **28,479 testing observations** accompanied by ground-truth binary anomaly labels (`test_label/machine-1-1.txt`).
- **Feature Dimensionality**: Each observation comprises **38 anonymized system performance metrics** capturing CPU load, memory footprint, disk I/O, network throughput, and TCP activity over time.
- **Role in Data Collection Strategy**: Proved that `IsolationForest` can model high-dimensional, real-world server telemetry and isolate genuine production incidents without requiring labeled anomalies during training.

### 2.2 Phase B: Live Containerized Microservice Telemetry (`collected_metrics.csv`)
To collect representative data tailored specifically to a live CI/CD canary deployment, a containerized microservice testbed was engineered using **Docker Compose**, **FastAPI**, **Prometheus**, **psutil**, and **Locust**:
1. **Target Microservices (`api_stable` and `api_canary`)**:
   - Built from a shared FastAPI codebase (`api/main.py`) exposing `/` (application endpoint) and `/metrics` (Prometheus telemetry endpoint).
   - Controlled via the `VERSION` environment variable:
     - `VERSION=stable` (Port `8000`): Simulates healthy production behavior where per-request latency is fast (`~10–18 ms`), memory remains stable (`~45.6 MB`), and CPU usage scales proportionally with incoming request volume.
     - `VERSION=canary_poisoned` (Port `8001`): Simulates a faulty canary deployment containing an algorithmic CPU regression, high response latency (`~300–900 ms`), and a bounded memory leak (`~258 MB`).
2. **Non-Stationary Traffic Generation (`load_tester/locustfile.py`)**:
   - Uses a custom Locust `LoadTestShape` (`RealisticTrafficShape`) that routes 80% of traffic to `stable` and 20% to `canary` while cycling every 60 seconds through realistic load regimes:
     - *00s–20s*: Normal diurnal wave (`18–45` concurrent users)
     - *20s–35s*: **Sudden Flash-Crowd User Influx** (`110` concurrent users, driving request rates above `115 req/s`)
     - *35s–50s*: Moderate steady traffic (`30` users)
     - *50s–60s*: Quiet off-peak trough (`8` users)
3. **Telemetry Scraping & Dataset Composition**:
   - The ML monitor (`ml_monitor/monitor.py`) polls `/metrics` every `1.5` seconds, accumulating **200 live observations** into a structured Pandas DataFrame (`collected_metrics.csv`):
     - **150 Baseline Training Samples (`stable`)**: Capturing four full 60-second traffic waves, including four flash-crowd user surges.
     - **30 Unseen Healthy Validation Samples (`stable_val`)**: Capturing a complete unseen traffic-surge wave on the healthy service to test false-positive resilience.
     - **20 Poisoned Canary Samples (`canary`)**: Capturing live telemetry after switching monitoring to the canary deployment.

| Telemetry Feature | Healthy Stable (`n = 180`, Mean ± SD) | Poisoned Canary (`n = 20`, Mean ± SD) | Physical Interpretation |
| :--- | :---: | :---: | :--- |
| **Request Rate (`requests_per_second`)** | `54.68 ± 32.79 req/s` | `18.49 ± 6.96 req/s` | Reflects dynamic 80/20 traffic split & user surges |
| **CPU Usage (`cpu_usage`)** | `34.05% ± 18.01%` | `67.43% ± 13.82%` | Stable spikes to `70%` only at `120 RPS`; Canary burns `67%` at `18 RPS` |
| **Memory Usage (`memory_usage`)** | `45.65 ± 0.40 MB` | `258.78 ± 6.18 MB` | Canary exhibits a ~5.7× memory footprint bloat |
| **Request Latency (`request_latency`)** | `0.0149 ± 0.0028 s` | `0.7237 ± 0.1472 s` | Canary exhibits a ~48.6× latency degradation |

---

## 3. Machine Learning Approach & Implementation Details

### 3.1 Rationale for Choosing Unsupervised Learning (`IsolationForest`)
A central requirement of this project was selecting a machine learning paradigm aligned with the underlying software engineering problem:
- **Why Unsupervised Anomaly Detection Instead of Supervised Classification?**
  In a real CI/CD pipeline, engineering teams possess abundant historical telemetry from the healthy `stable` version currently running in production, but **zero labeled training examples** of the unknown bug that a new `canary` commit might introduce tomorrow. A supervised classifier (such as Random Forest or Logistic Regression) can only recognize failure modes it was explicitly trained on. Unsupervised one-class anomaly detection learns the boundary of "healthy normal operation" exclusively from `stable` baseline data and flags any significant deviation.
- **Why `IsolationForest` Specifically?**
  Proposed by Liu et al. (2008), Isolation Forest is an ensemble tree-based algorithm that explicitly isolates anomalies rather than profiling normal density:
  1. It builds `n_estimators = 100` random binary trees (**iTrees**), each trained on a random subsample of up to $\psi = 256$ baseline points.
  2. At each internal node, a feature is selected at random and split at a random value between its minimum and maximum.
  3. **Normal points** lie within dense clusters (such as normal load or proportional high-RPS traffic surges) and require **many random splits** (a deep tree path length $h(x)$) to isolate.
  4. **Anomalous points** (such as the poisoned canary) lie in sparse regions of the feature space and are isolated in **very few splits** near the root of the tree (a short path length $h(x)$).
  5. It operates in $O(n \log \psi)$ training time and sub-millisecond inference time, making it ideal for real-time 1.5–2.0 second CI/CD polling loops.

### 3.2 Implementation Pipeline (`ml_monitor/monitor.py`)

#### Step 1: Data Preprocessing & Feature Engineering (`preprocess_data()`)
Raw system metrics collected over HTTP cannot be fed blindly into a production ML pipeline. The preprocessing module performs three operations:
1. **Sanitization & Imputation**: Replaces any infinite readings (`±inf`) with `NaN`, applies forward-fill (`ffill`) followed by median imputation to handle transient network scrape hiccups, and clips physically impossible negative readings to `0.0`.
2. **Traffic-Aware Feature Engineering (`cpu_per_rps`)**: To resolve the Traffic Confounding Problem, the pipeline engineers a derived ratio feature:
   $$\text{cpu\_per\_rps} = \frac{\text{cpu\_usage}}{\max(\text{requests\_per\_second}, 1.0)}$$
   During a healthy user influx (`120 RPS`, `68% CPU`), `cpu_per_rps` remains low (`~0.57% per RPS`). During a canary regression (`18 RPS`, `68% CPU`), `cpu_per_rps` jumps above `3.7% per RPS`, providing a strong discriminative signal alongside raw CPU, Memory, and Latency.
3. **Feature Standardization (`StandardScaler`)**: Because the 5 features have incompatible units (`req/s`, `%`, `%/req`, `MB`, and `seconds`), a `scikit-learn` `Pipeline` applies $z$-score normalization ($\mu = 0, \sigma = 1$) fitted exclusively on the stable baseline data.

#### Step 2: Model Training & Hyperparameter Rationale (`train_baseline()`)
The pipeline fits `IsolationForest` on the 150 baseline `stable` samples with the following hyperparameters:
- `n_estimators = 100`: Ensemble of 100 isolation trees to ensure stable path-length convergence.
- `max_samples = "auto"` ($\min(256, n)$): Subsampling size recommended by the original Isolation Forest authors to prevent swamping and masking.
- `contamination = 0.01`: Estimates that at most 1% of baseline training observations represent extreme statistical tail noise, setting the **Decision Boundary ($s(x) = 0$)** at the 1st percentile of the baseline score distribution.
- `random_state = 42`: Guarantees deterministic reproducibility across runs.

#### Step 3: Real-Time Canary Evaluation & Debounced Rollback (`monitor_canary()`)
For each incoming telemetry sample $x$, the model computes:
- Continuous anomaly score $s(x) = \text{model.decision\_function}(x)$, where **$s(x) \ge 0$ indicates normal operation (`+1`)** and **$s(x) < 0$ indicates an anomaly (`-1`)**.
- **Consecutive-Anomaly Debouncing**: Rather than aborting a deployment on a single `-1` tick (which could be caused by a momentary OS garbage-collection pause), the monitor tracks `consecutive_anomalies` and triggers a full deployment rollback only when anomalies persist across consecutive polling intervals.

---

## 4. Validation Results & System Effectiveness

### 4.1 Phase A Validation: Offline Real-World SMD Benchmark
In `canaryrollbackML/smd_analysis.ipynb`, the `IsolationForest` model was trained on the 28,479 healthy time steps of `ServerMachineDataset/train/machine-1-1.txt` and evaluated against ground-truth incident labels on `test/machine-1-1.txt`:
- **Point-Wise Precision**: `0.297` (`29.7%`)
- **Point-Wise Recall**: `0.409` (`40.9%`)
- **Point-Wise F1-Score**: `0.344` (`34.4%`)

**Analysis of Offline Results**: On raw 38-dimensional real-world server telemetry without feature engineering or window smoothing, point-wise metrics are conservative because ground-truth incident labels mark entire contiguous incident windows as anomalous, whereas `IsolationForest` triggers primarily on the sharpest metric peaks within each incident window. Crucially, visual inspection of the first 3,000 time steps confirmed that every major server incident spike was successfully flagged by red rollback trigger points.

> **[INSERT IMAGE 1 HERE: Jupyter Notebook Graph from `canaryrollbackML/smd_analysis.ipynb` ("Dynamic CI/CD Rollback Triggers on Live Server Telemetry")]**

---

### 4.2 Phase B Validation: Live Containerized CI/CD Pipeline (`collected_metrics.csv`)
The complete traffic-aware pipeline was validated live in Docker across 200 polling steps (`150` baseline training + `30` unseen stable traffic-surge validation + `20` poisoned canary):

```text
=================================================================
 Phase 3: Model Evaluation & Live Validation Summary
=================================================================
 Total Telemetry Samples : 200 (Healthy Stable=180, Poisoned Canary=20)
 True Positives  (TP)    :   20 | False Positives (FP) :    2
 True Negatives  (TN)    :  178 | False Negatives (FN) :    0
 Precision               : 0.9091 (90.91%)
 Recall (Sensitivity)    : 1.0000 (100.00%)
 F1-Score                : 0.9524 (95.24%)
 False Positive Rate     : 0.0111 (1.11%)
=================================================================
```

#### Key Validation Findings:
1. **100% Resilience to Unseen User Traffic Surges (Steps 150–180)**:
   During Phase 2A (`stable_val`), Locust generated a sudden flash-crowd surge of 110 users (`115 RPS`), causing `api_stable` CPU usage to spike to **`68%`**. Because request rate scaled proportionally and memory (`45.6 MB`) and latency (`15 ms`) remained normal, the model classified **30 out of 30 unseen surge samples (`100%`) as Normal (`+1`)**, completely avoiding false rollbacks.
2. **100% Sensitivity to Canary Regressions (Steps 180–200)**:
   Immediately upon switching to `api_canary` at step 180, the decision score $s(x)$ plunged from `+0.14` to `-0.075` (well below the `score = 0` decision boundary). All **20 out of 20 canary samples (`Recall = 100.0%`)** were flagged as anomalies (`-1`), initiating rollback alerts on every single poll.
3. **Elimination of False Rollbacks via Consecutive Debouncing**:
   The only 2 false positives out of 180 healthy samples (`FPR = 1.11%`, matching the theoretical `contamination = 0.01` calibration) occurred as isolated single-step ticks during baseline training (steps 95 and 116) and never occurred consecutively. Consequently, deployment-level rollback accuracy was **100%**.

> **[INSERT IMAGE 2 HERE: Live Pipeline 5-Panel Validation Figure from `ml_monitor/live_validation_results.png` ("Traffic-Aware CI/CD Canary Validation — Dynamic User Influxes vs. Poisoned Canary")]**

---

## 5. Suggestions for Future Improvements & Further Research

While the current system successfully automates canary anomaly detection and resolves traffic confounding, several enhancements could be pursued in future iterations:
1. **Temporal Rolling-Window Derivative Features**: Adding 30-second rolling slopes (e.g., $\Delta\text{Memory}/\Delta t$ and moving-average latency variance) would allow the model to detect slow, low-grade memory leaks over multi-hour canary bake periods before absolute memory crosses baseline variance.
2. **Expansion to All Four SRE Golden Signals**: Extending the Prometheus scrape payload to include **HTTP `5xx` error rates** and **99th-percentile (`p99`) histogram tail latency** (`app_request_duration_seconds_bucket`) would enable detection of functional application errors that do not affect CPU or memory.
3. **Closed-Loop Service Mesh Integration (Automated Traffic Shifting)**: Connecting `monitor_canary()`'s rollback trigger directly to an **NGINX**, **Envoy**, or **Kubernetes Argo Rollouts** webhook would allow the monitor to automatically dial canary traffic weights from `20%` down to `0%` and terminate the faulty container without human intervention.
4. **Side-by-Side Differential Canary Scoring**: Computing real-time pairwise metric ratios between `api_canary` and `api_stable` at the exact same timestamp ($\text{Latency}_{\text{canary}}(t) / \text{Latency}_{\text{stable}}(t)$) would cancel out global environment shifts (such as shared database slowdowns) automatically.
