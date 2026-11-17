import collections
import math
import os
import random
import threading
import time
from typing import Deque, List

import psutil
from fastapi import FastAPI, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

app = FastAPI(title="Dynamic Canary Rollback Target API")

VERSION = os.getenv("VERSION", "stable")
PROCESS = psutil.Process(os.getpid())

# Prime psutil CPU measurement so subsequent calls return non-zero readings
PROCESS.cpu_percent(interval=None)
psutil.cpu_percent(interval=None)

# Prometheus Metrics
REQUEST_RATE = Gauge(
    "app_requests_per_second",
    "Current incoming request rate (requests/sec) over a 5-second sliding window",
)
CPU_USAGE = Gauge(
    "app_cpu_usage_percent",
    "Current application CPU usage percentage",
)
MEMORY_USAGE = Gauge(
    "app_memory_usage_mb",
    "Current application RSS memory usage in MB",
)
REQUEST_LATENCY = Gauge(
    "app_request_latency_seconds",
    "Rolling average request processing latency in seconds",
)
REQUEST_LATENCY_HIST = Histogram(
    "app_request_duration_seconds",
    "Histogram of request processing latency in seconds",
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5),
)
REQUEST_COUNT = Counter(
    "app_requests_total",
    "Total number of requests received",
    ["version", "endpoint"],
)

# Sliding window of recent request timestamps & latencies (5-second window)
_WINDOW_SECONDS = 5.0
_REQ_LOCK = threading.Lock()
_RECENT_REQUESTS: Deque[tuple[float, float]] = collections.deque()

# Bounded memory buffer to simulate a memory leak / bloat in canary_poisoned
# Capped at ~80 MB so the container does not get OOM-killed by Docker
_LEAK_BUFFER: List[bytes] = []
_MAX_LEAK_CHUNKS = 80  # 80 * 1 MB = 80 MB max retained


def _simulate_cpu_burn(duration_sec: float) -> None:
    """Perform CPU-bound arithmetic for `duration_sec` seconds."""
    end_time = time.perf_counter() + duration_sec
    acc = 0.0
    while time.perf_counter() < end_time:
        acc += math.sqrt(random.random() * 10000.0)


def _prune_and_get_window_stats(now: float) -> tuple[float, float]:
    """
    Removes request records older than `_WINDOW_SECONDS` and returns
    (requests_per_second, avg_latency_seconds).
    """
    cutoff = now - _WINDOW_SECONDS
    with _REQ_LOCK:
        while _RECENT_REQUESTS and _RECENT_REQUESTS[0][0] < cutoff:
            _RECENT_REQUESTS.popleft()
        count = len(_RECENT_REQUESTS)
        rps = count / _WINDOW_SECONDS
        avg_lat = (
            sum(lat for _, lat in _RECENT_REQUESTS) / count if count > 0 else 0.0
        )
    return rps, avg_lat


@app.get("/")
def read_root():
    start_time = time.perf_counter()

    if VERSION == "canary_poisoned":
        # 1. Artificial high latency regression
        time.sleep(random.uniform(0.30, 0.70))

        # 2. Disproportionate CPU burn per request
        _simulate_cpu_burn(random.uniform(0.035, 0.070))

        # 3. Memory leak (bounded at 80 MB) + transient allocation
        if len(_LEAK_BUFFER) < _MAX_LEAK_CHUNKS:
            _LEAK_BUFFER.append(os.urandom(1024 * 1024))
        _transient_spike = [random.random() for _ in range(250_000)]
    else:
        # Healthy stable release: light per-request CPU work (~2-4ms) + fast I/O (~3-10ms)
        # Under normal traffic CPU is low (~10%), but during a flash-crowd user influx
        # CPU scales up proportionally with RPS while latency & memory remain healthy!
        _simulate_cpu_burn(random.uniform(0.002, 0.004))
        time.sleep(random.uniform(0.003, 0.010))

    now = time.perf_counter()
    latency = now - start_time

    with _REQ_LOCK:
        _RECENT_REQUESTS.append((now, latency))

    REQUEST_LATENCY.set(round(latency, 4))
    REQUEST_LATENCY_HIST.observe(latency)
    REQUEST_COUNT.labels(version=VERSION, endpoint="/").inc()

    return {
        "message": f"Hello from the {VERSION} service!",
        "version": VERSION,
        "latency_seconds": round(latency, 4),
    }


@app.get("/metrics")
def metrics():
    REQUEST_COUNT.labels(version=VERSION, endpoint="/metrics").inc()

    now = time.perf_counter()
    rps, avg_lat = _prune_and_get_window_stats(now)

    # Measure actual process CPU & RSS memory via psutil
    proc_cpu = PROCESS.cpu_percent(interval=None)
    sys_cpu = psutil.cpu_percent(interval=None)
    cpu_val = max(proc_cpu, sys_cpu)

    mem_mb = PROCESS.memory_info().rss / (1024 * 1024)

    if VERSION == "canary_poisoned":
        if cpu_val < 20.0:
            cpu_val += random.uniform(40.0, 65.0)
        if avg_lat == 0.0:
            avg_lat = random.uniform(0.40, 0.70)
        if len(_LEAK_BUFFER) == 0:
            mem_mb += random.uniform(55.0, 85.0)
    else:
        # Ensure CPU smoothly tracks real RPS surges (e.g., 10% at low load up to 65% during user influx)
        expected_load_cpu = 4.0 + (rps * 0.55) + random.uniform(-1.2, 1.2)
        cpu_val = max(cpu_val, expected_load_cpu)
        cpu_val = min(cpu_val, 95.0)
        if avg_lat == 0.0:
            avg_lat = random.uniform(0.006, 0.014)
        else:
            # Slight realistic queuing jitter at very high RPS, still well under canary regression
            avg_lat = avg_lat + (rps * 0.00008)

    REQUEST_RATE.set(round(rps, 2))
    CPU_USAGE.set(round(cpu_val, 3))
    MEMORY_USAGE.set(round(mem_mb, 3))
    REQUEST_LATENCY.set(round(avg_lat, 4))

    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
