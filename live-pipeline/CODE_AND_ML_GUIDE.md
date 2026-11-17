# Complete Beginner's Guide to the Code & Machine Learning Techniques

This document explains **every concept, algorithm, and file** in this project from the ground up, written for someone with **zero prior background in Machine Learning**.

---

## Part 1: The Big Picture — What Are We Building and Why?

### 1. What is a "Canary Deployment"?
Imagine you run a popular banking app used by 1 million people, and your developers just wrote a new update (**Version 2.0**).
- If you replace the old app (**Version 1.0**) with Version 2.0 for all 1 million users at once, and Version 2.0 has a hidden bug that freezes the server, **all 1 million users crash**.
- Instead, companies use a **Canary Deployment** (named after the "canary in a coal mine"):
  - Keep **80% of users** on the safe, proven **Stable Version** (`api_stable`).
  - Send **20% of users** to the new **Canary Version** (`api_canary`).
  - Watch the Canary closely for a few minutes. If it acts healthy, move everyone to it. If it starts acting sick, immediately **Roll Back** (send those 20% of users back to the Stable version and shut down the Canary).

### 2. Why Do We Need Machine Learning Instead of Simple `if/else` Rules?
You might wonder: *"Why not just write an `if` statement like `if CPU > 60%: rollback()`?"*

Because **normal user traffic changes throughout the day**:
- At 3:00 AM, only 8 people are using the app, so CPU usage is `10%`.
- At 6:00 PM, a huge wave of 110 people logs in at the same time. Even on a **100% healthy server**, CPU usage will jump to **`70%`** just because it is serving more customers!
- A simple `if CPU > 60%` rule would panic during the 6:00 PM rush and falsely shut down a healthy server.

Instead, we need an algorithm that looks at **all metrics together** (Traffic Rate, CPU, Memory, and Speed/Latency) and understands the difference between:
1. **A Healthy Rush Hour**: *"CPU is 70%, BUT Traffic is 120 requests/sec, Memory is normal (45 MB), and responses are fast (0.015 seconds). Everything is fine!"*
2. **A Broken Canary Bug**: *"Traffic is only 18 requests/sec, YET CPU is 67%, Memory bloated to 258 MB, and responses take 0.72 seconds. This version is broken—ROLL BACK!"*

---

## Part 2: Machine Learning Concepts Explained Simply

### 1. Supervised vs. Unsupervised Machine Learning
In Machine Learning, there are two main ways a computer learns:
- **Supervised Learning (Learning with Answer Keys)**: You show the computer 5,000 photos labeled *"Cat"* and 5,000 photos labeled *"Dog"*. It learns the difference.
  - *Why this DOES NOT work for our project*: When developers push new code tomorrow, we don't have 5,000 examples of how tomorrow's brand-new bug will break the server! We only have data from the **healthy** server running today.
- **Unsupervised Anomaly Detection (Learning What "Normal" Looks Like)**: We only feed the computer data from the **healthy (`stable`) server**. The algorithm learns the shape of "normal behavior." Later, if a new data point looks completely different from everything it saw during training, it flags it as an **Anomaly** (an outlier).

### 2. How Does an "Isolation Forest" (`IsolationForest`) Work?
Despite the fancy name, **Isolation Forest** is built on a very simple idea: **It is much easier to separate an weird outlier from a crowd than it is to separate a normal person standing in the middle of a crowd.**

#### The Party Analogy:
Imagine 100 people are standing tightly packed in the center of a room (normal data points), and **1 person** is standing all alone in the far corner (an anomaly).
If you close your eyes and start putting up random walls across the room:
- Tofence off the **1 person in the corner** by themselves, it only takes **1 or 2 random walls**.
- To fence off a specific person standing **in the middle of the tight crowd**, you have to build **10 or 15 walls** before they are standing alone.

#### How the Algorithm Does This Mathematically:
1. **A "Tree" (`iTree`)**: The algorithm picks a random metric (say, `Memory Usage`) and picks a random split number (say, `150 MB`). Everyone below `150 MB` goes left; everyone above `150 MB` goes right. It keeps splitting randomly until every data point is isolated in its own box.
2. **A "Forest" (`n_estimators = 100`)**: Because one random tree might get lucky or unlucky, the algorithm builds **100 random trees** (a "forest") and averages how many splits it took to isolate a point:
   - **Many splits needed (Long path down the tree)** $\rightarrow$ The point is right in the middle of normal data $\rightarrow$ **Prediction = `+1` (Normal)**.
   - **Very few splits needed (Short path near the top of the tree)** $\rightarrow$ The point is far away from normal data $\rightarrow$ **Prediction = `-1` (Anomaly)**.

### 3. What is the "Decision Boundary (`score = 0`)"?
In addition to outputting `+1` (Normal) or `-1` (Anomaly), `IsolationForest` gives each data point a continuous **Health/Anomaly Score** using `model.decision_function()`:
- **Score $\ge 0$ (Above the red dotted line on our graph)**: Normal behavior (`+1`). The higher above zero, the more normal it is.
- **Score $< 0$ (Below the red dotted line on our graph)**: Anomalous behavior (`-1`). The further below zero, the more extreme the anomaly is.
- Therefore, **`score = 0`** is called the **Decision Boundary**—it is the exact borderline where the algorithm switches its decision from *"Safe"* to *"Rollback!"*

### 4. What is `contamination = 0.01`?
Even a healthy server occasionally has a random 1-second hiccup (for example, when the operating system runs background cleanup). When we set `contamination = 0.01` (`1%`), we tell the Isolation Forest:
> *"Assume about 99% of the training data is truly normal, and the most extreme 1% of points are random hiccups. Draw the `score = 0` Decision Boundary right at that 99% border."*

### 5. What is Data Preprocessing & `StandardScaler`?
Look at the numbers our server reports:
- **Request Rate**: `10` to `120` requests/sec
- **Memory Usage**: `45` to `260` Megabytes
- **Request Latency**: `0.01` to `0.75` seconds

Notice that Latency is a tiny decimal (`0.01`), while Memory is a big number (`260`). To make sure the pipeline treats all metrics cleanly regardless of their measurement units (`%`, `MB`, or `seconds`), we use **`StandardScaler`**:
- It converts every metric into *"How many standard deviations away from average is this number?"* (called a **$z$-score**), where `0.0` is the average of the healthy server.

### 6. What Are Precision, Recall, F1-Score, and the Confusion Matrix?
To grade how well our ML model performed, we compare its predictions (`+1` or `-1`) against reality (`Stable` vs. `Canary`):
- **True Positive (`TP = 20`)**: The server was the broken Canary, and the model **correctly** flagged it as an Anomaly (`-1`).
- **True Negative (`TN = 178`)**: The server was the healthy Stable version, and the model **correctly** said it was Normal (`+1`).
- **False Positive (`FP = 2`)**: *False Alarm.* The server was healthy, but the model flagged a single tick as an Anomaly (`-1`).
- **False Negative (`FN = 0`)**: *Missed Bug.* The server was broken, but the model failed to catch it. (We got `0`—our model didn't miss a single one!)

From those four numbers, we calculate three percentage grades (from `0.0` to `1.0`):
1. **Recall (`1.0000` or `100%`)**: *"Out of all the actual broken Canary samples, what percentage did the model catch?"*
   $$\text{Recall} = \frac{TP}{TP + FN} = \frac{20}{20 + 0} = 100\%$$
2. **Precision (`0.9091` or `90.91%`)**: *"When the model raised an Anomaly alarm, what percentage of those alarms were actually the broken Canary?"*
   $$\text{Precision} = \frac{TP}{TP + FP} = \frac{20}{20 + 2} = 90.91\%$$
3. **F1-Score (`0.9524` or `95.24%`)**: The balanced average (harmonic mean) of Precision and Recall.

---

## Part 3: File-by-File Code Walkthrough

### 1. The Offline Real-World Notebook (`canaryrollbackML/smd_analysis.ipynb`)
Before building our own servers, this Jupyter Notebook tested `IsolationForest` on a real-world public dataset (**Server Machine Dataset**, `machine-1-1.txt`):
- **Cell 1**: Loads `train/machine-1-1.txt` (28,479 rows of healthy server data across 38 metrics), trains `IsolationForest(n_estimators=100, contamination=0.01)`, and predicts anomalies on `test/machine-1-1.txt`.
- **Cell 2**: Compares the predictions against the true incident labels (`test_label/machine-1-1.txt`) and prints Precision, Recall, and F1-Score.
- **Cell 3**: Plots the first 3,000 time steps in blue and overlays red dots wherever the model would trigger a rollback.

---

### 2. The Target Web Server (`live-pipeline/api/main.py` & `api/Dockerfile`)
This file is a Python web server built with **FastAPI** that simulates our application in production. Because we run two copies of this file in Docker, it checks an environment variable called `VERSION`:
- **`VERSION = os.getenv("VERSION", "stable")`**:
  - Container 1 (`api_stable` on port `8000`) runs with `VERSION=stable`.
  - Container 2 (`api_canary` on port `8001`) runs with `VERSION=canary_poisoned`.
- **Endpoint 1 (`@app.get("/")`)**:
  - Simulates a user visiting the homepage.
  - If `VERSION == "stable"`: Does a tiny, normal amount of CPU work (`2–4 ms`) and replies in `~15 milliseconds`.
  - If `VERSION == "canary_poisoned"`: Simulates a buggy software update! It pauses for `0.30–0.70 seconds` (`time.sleep()`), burns heavy CPU in a `math.sqrt()` loop, and appends 1 MB of random bytes to a list (`_LEAK_BUFFER`) to simulate a memory leak (capped at 80 MB so it doesn't crash your computer).
- **Endpoint 2 (`@app.get("/metrics")`)**:
  - Uses the `psutil` library (which reads actual CPU % and RAM usage from the operating system) and `prometheus_client` to publish 4 live gauges in Prometheus format:
    1. `app_requests_per_second`: How many requests hit `/` per second over the last 5 seconds.
    2. `app_cpu_usage_percent`: Current CPU usage (`%`).
    3. `app_memory_usage_mb`: Current RAM usage (`MB`).
    4. `app_request_latency_seconds`: Average response time (`seconds`).

---

### 3. The Virtual User Load Generator (`live-pipeline/load_tester/locustfile.py`)
A server sitting idle generates no realistic data. We use **Locust**, an industry-standard load-testing tool that spawns simulated users who continuously click on `/`:
- **`TrafficUser(HttpUser)`**:
  - Has two tasks weighted `@task(4)` and `@task(1)` (a 4-to-1 ratio = **80% / 20% split**).
  - 80% of requests go to `api_stable` (`http://api_stable:8000/`), and 20% go to `api_canary` (`http://api_canary:8000/`).
- **`RealisticTrafficShape(LoadTestShape)`**:
  - Automatically changes the number of virtual users every second in a repeating **60-second wave**:
    - Seconds `0–20`: Smooth sine wave of `18 to 45` users.
    - Seconds `20–35`: **Sudden User Influx!** Jumps to **`110` users** at once, causing request rate (`RPS`) and `CPU` on the server to spike.
    - Seconds `35–50`: Calms down to `30` users.
    - Seconds `50–60`: Quiet period of only `8` users.

---

### 4. The ML Anomaly Monitor (`live-pipeline/ml_monitor/monitor.py`)
This script is the centerpiece of the live pipeline. It runs in three stages:

1. **`fetch_metrics(metrics_url)`**:
   - Sends an HTTP `GET` request to `/metrics` every `1.5` seconds, reads the Prometheus text lines, and extracts `requests_per_second`, `cpu_usage`, `memory_usage`, and `request_latency` into a Python dictionary.
2. **`preprocess_data(df)`**:
   - Cleans any missing/corrupted numbers.
   - **Feature Engineering**: Creates a 5th column called **`cpu_per_rps`** (`cpu_usage / requests_per_second`). This tells the model how much CPU each individual request is costing.
3. **`train_baseline(df)`**:
   - Takes the `150` data points collected from `api_stable` during Phase 1.
   - Builds a `scikit-learn` `Pipeline` that first normalizes the columns with `StandardScaler()` and then trains `IsolationForest(n_estimators=100, contamination=0.01)`.
4. **`monitor_canary(model, data_point)`**:
   - Feeds each new data point into `model.predict()` and `model.decision_function()`.
   - If `prediction == -1`, prints:
     `[ALERT] ... -> ANOMALY DETECTED: INITIATING ROLLBACK`
   - If `prediction == 1`, prints:
     `[OK] ... -> Normal` (and adds `[Traffic Surge Handled]` when RPS is high!).
5. **The 3-Phase Execution Loop (`run_pipeline()`)**:
   - **Phase 1 (Steps 1–150)**: Scrapes `api_stable` across 4 traffic waves and trains the `IsolationForest`.
   - **Phase 2A (Steps 151–180)**: Continues watching `api_stable` during a brand-new 110-user traffic surge to prove the trained model **does not** trigger a false rollback during rush hour.
   - **Phase 2B (Steps 181–200)**: Switches to watching `api_canary`. Counts `consecutive_anomalies` so a single random blip won't cause a rollback, but sustained canary failures trigger `[ROLLBACK TRIGGERED]`.
   - **Phase 3 (`evaluate_model()`)**: Calculates Precision, Recall, F1-Score, False Positive Rate, and saves everything to `collected_metrics.csv`.

---

### 5. The Evaluation & Graph Generator (`live-pipeline/ml_monitor/evaluate_model.py`)
- Reads `collected_metrics.csv` after a run.
- Prints a clean comparison table (`Mean ± Standard Deviation`) comparing Healthy Stable vs. Poisoned Canary across all metrics.
- Uses `matplotlib` to draw the 5-panel chart (`ml_monitor/live_validation_results.png`), plotting Request Rate, CPU, Memory, Latency, and the Isolation Forest Decision Score over time, with red dots marking every `-1` anomaly prediction.

---

### 6. Docker Orchestration (`live-pipeline/docker-compose.yml`)
Instead of manually installing and configuring servers on different ports, `docker-compose.yml` launches the entire microservice environment in isolated containers:
- **`api_stable`**: Runs `./api` on port `8000` with `VERSION=stable`.
- **`api_canary`**: Runs the exact same `./api` code on port `8001`, but with `VERSION=canary_poisoned`.
- **`load_tester`**: Runs the official `locustio/locust` image using `load_tester/locustfile.py` to send live traffic to both servers.
- **`ml_monitor`**: Containerized version of the ML monitor.
