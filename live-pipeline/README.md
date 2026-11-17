# Dynamic CI/CD Canary Rollback Using Unsupervised Machine Learning (`IsolationForest`)

## 1. Overview of the Software Engineering Problem Addressed
In modern Continuous Integration and Continuous Deployment (CI/CD) pipelines, **Canary Releases** mitigate deployment risk by routing a small subset of live user traffic (e.g., 20%) to a newly deployed version (`canary`) while the majority of traffic (80%) continues to be served by the proven production version (`stable`).

Traditionally, canary health checks rely on **static, hand-crafted threshold rules** (e.g., `CPU > 60%` or `Latency > 500ms`). However, static thresholds suffer from two critical software engineering limitations:
1. **The Traffic Confounding Problem (False Rollbacks on User Surges)**: When a legitimate influx of users hits a healthy server, CPU utilization naturally spikes (e.g., from `10%` to `70%`). A static threshold (`CPU > 60%`) or a traffic-blind anomaly detector misinterprets a healthy user rush as a broken release and triggers a false rollback.
2. **Missed Multi-Dimensional Regressions**: Subtle software regressions—such as a memory leak paired with elevated tail latency and disproportionate CPU burn per request—require evaluating telemetry **multivariately** relative to active request load.

This project automates **Traffic-Aware CI/CD Canary Health Monitoring and Rollback** using an unsupervised **Isolation Forest** pipeline that learns the joint multivariate relationship between **Request Rate (`RPS`)**, **CPU Usage (`%`)**, **CPU per Request (`CPU/RPS`)**, **Memory Usage (`MB`)**, and **Request Latency (`s`)**, distinguishing benign user traffic surges from true canary regressions.

---

## 2. Data Collection Methodology
To satisfy both **real-world generalization** and **live software engineering validation**, a two-phase data collection strategy was designed:

### Phase A: Offline Real-World Benchmark (Server Machine Dataset — SMD)
- **Source**: Publicly available **Server Machine Dataset (`machine-1-1.txt`)** in [smd_analysis.ipynb](../canaryrollbackML/smd_analysis.ipynb).
- **Composition**: `28,479` baseline training time steps and `28,479` labeled test time steps across `38` multivariate server performance metrics.
- **Purpose**: Proves that the `IsolationForest` algorithm detects real-world, multivariate server anomalies prior to building the live microservice infrastructure.

### Phase B: Live Containerized Microservice Telemetry Under Dynamic Traffic Waves ([collected_metrics.csv](collected_metrics.csv))
- **Source**: Custom containerized FastAPI microservice ([api/main.py](api/main.py)) instrumented with `prometheus_client` and `psutil`, scraped at `1.5s` intervals under a **non-stationary Locust load profile** ([load_tester/locustfile.py](load_tester/locustfile.py) `RealisticTrafficShape`) that cycles every 60 seconds through:
  - **Normal Diurnal Wave** (`18–45` users)
  - **Sudden Flash-Crowd User Influx** (`110` concurrent users, driving `RPS` up to `120 req/s` and healthy `stable` CPU up to `70%`)
  - **Moderate Load** (`30` users) & **Off-Peak Quiet Trough** (`8` users)
- **Dataset Size**: `200` live telemetry observations split across three stages:
  1. `150` **Baseline Training Samples (`stable`)**: Capturing 4 full traffic-surge cycles.
  2. `30` **Unseen Healthy Traffic-Surge Validation Samples (`stable_val`)**: Capturing a full unseen 110-user influx wave on the healthy service.
  3. `20` **Poisoned Canary Samples (`canary`)**: Capturing telemetry from `VERSION=canary_poisoned`.

| Telemetry Metric | Healthy Stable (`n = 180`, Mean ± SD) | Poisoned Canary (`n = 20`, Mean ± SD) | Behavior Summary |
| :--- | :---: | :---: | :--- |
| **Request Rate (`RPS`)** | `54.68 ± 32.79 req/s` (Range: `8–124`) | `18.49 ± 6.96 req/s` | Dynamic 80/20 canary traffic split |
| **CPU Usage (`%`)** | `34.05% ± 18.01%` (Spikes to `70%` at high RPS) | `67.43% ± 13.82%` | Canary burns high CPU even at low RPS |
| **Memory Usage (`MB`)** | `45.65 ± 0.40 MB` | `258.78 ± 6.18 MB` | Canary exhibits memory leak/bloat (~5.7×) |
| **Request Latency (`s`)** | `0.0149 ± 0.0028 s` (~15 ms) | `0.7237 ± 0.1472 s` (~724 ms) | Canary exhibits ~48.6× latency degradation |

---

## 3. Machine Learning Approach & Implementation Details

### 3.1 Rationale for Choosing Unsupervised `IsolationForest`
- **Absence of Labeled Failures in CI/CD**: Production teams have abundant telemetry from the healthy `stable` release, but zero labeled examples of novel bugs a future `canary` commit might introduce.
- **Multivariate Isolation**: By partitioning standardized features randomly across `n_estimators = 100` isolation trees, `IsolationForest` learns that high CPU (`~70%`) paired with high request rate (`~120 RPS`), flat memory (`~45.6 MB`), and low latency (`~15 ms`) lies inside the normal cluster, whereas high CPU/memory/latency at low RPS lies in sparse space and is isolated in very few splits.

### 3.2 Implementation Pipeline ([ml_monitor/monitor.py](ml_monitor/monitor.py))
1. **Data Preprocessing & Traffic-Aware Feature Engineering (`preprocess_data()`)**:
   - Sanitizes missing or infinite readings using forward-fill (`ffill`) and median imputation, and clips negative values to `0.0`.
   - Engineers **`cpu_per_rps`** $=\text{CPU Usage} / \max(\text{RPS}, 1.0)$ to capture CPU cost per unit of traffic.
   - Standardizes all 5 features (`requests_per_second`, `cpu_usage`, `cpu_per_rps`, `memory_usage`, `request_latency`) via **`StandardScaler` ($z$-score normalization)** inside a `scikit-learn` `Pipeline`.
2. **Decision Boundary (`score = 0`) & `contamination = 0.01`**:
   - `model.decision_function(x)` outputs a continuous anomaly score $s(x)$ where **$s(x) = 0$ is the Decision Boundary**:
     - **$s(x) \ge 0 \implies \text{Prediction } +1$ (Normal)**: The data point requires many random tree splits to isolate (dense healthy behavior).
     - **$s(x) < 0 \implies \text{Prediction } -1$ (Anomaly)**: The data point is isolated near the root of the trees (anomalous behavior).
   - Setting `contamination = 0.01` places the `score = 0` threshold at the 1st percentile of the baseline training distribution.
3. **Consecutive Anomaly Debouncing**:
   - Requires `k` consecutive `-1` predictions before triggering a production rollback, filtering out isolated single-tick statistical outliers (`FP = 2` single ticks during training) while catching sustained canary regressions (`20 / 20` consecutive `-1` predictions).

---

## 4. Validation Results & System Effectiveness

### 4.1 Offline Validation on Real-World SMD Dataset (`canaryrollbackML/smd_analysis.ipynb`)
- Validated on `machine-1-1` (`28,479` test points across 38 metrics), achieving **Precision = `0.297`**, **Recall = `0.409`**, **F1-Score = `0.344`**.

### 4.2 Live Traffic-Aware CI/CD Pipeline Validation ([ml_monitor/live_validation_results.png](ml_monitor/live_validation_results.png))
Across `200` live telemetry observations (`150` baseline training + `30` unseen stable traffic-surge validation + `20` poisoned canary):
- **Unseen Traffic-Surge Validation (Steps 150–180)**: When hit by an unseen flash-crowd surge (`115 RPS`, `68% CPU`), the model recorded **`0` False Positives (`30 / 30` classified as Normal `+1`)**.
- **Canary Regression Detection (Steps 180–200)**: **`20 / 20` canary samples** were immediately flagged as anomalies (`Recall = 1.0000`).
- **Overall Point-Wise Metrics**:
  - **Precision**: `0.9091` (`90.91%`)
  - **Recall (Sensitivity)**: `1.0000` (`100.00%`)
  - **F1-Score**: `0.9524` (`95.24%`)
  - **False Positive Rate (FPR)**: `0.0111` (`1.11%` — matching the theoretical `contamination = 0.01` setting, with **0% false rollbacks** under consecutive-window debouncing).

---

## 5. Suggestions for Future Improvements & Further Research
1. **Multivariate Rolling-Window Derivatives**: Include 30-second rolling slopes ($\Delta\text{Memory}/\Delta t$) to detect slow multi-hour memory leaks.
2. **Additional Golden Signals**: Incorporate HTTP `5xx` error rates and `p99` tail latency quantiles from Prometheus histograms.
3. **Automated Service-Mesh Rollback Webhook**: Connect `monitor_canary()` directly to Kubernetes **Argo Rollouts** or **NGINX/Envoy** to shift traffic weights to `0%` automatically upon rollback.
