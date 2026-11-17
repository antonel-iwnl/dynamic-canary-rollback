import math
import os
import random
import time
from typing import List

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
    "Most recent request processing latency in seconds",
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

# Bounded memory buffer to simulate a memory leak / bloat in canary_poisoned
# Capped at ~80 MB so the container does not get OOM-killed by Docker
_LEAK_BUFFER: List[bytes] = []
_MAX_LEAK_CHUNKS = 80  # 80 * 1 MB = 80 MB max retained


def _simulate_cpu_burn(duration_sec: float = 0.04) -> None:
    """Perform CPU-bound arithmetic for `duration_sec` seconds."""
    end_time = time.perf_counter() + duration_sec
    acc = 0.0
    while time.perf_counter() < end_time:
        acc += math.sqrt(random.random() * 10000.0)


@app.get("/")
def read_root():
    start_time = time.perf_counter()

    if VERSION == "canary_poisoned":
        # 1. Artificial high latency
        sleep_time = random.uniform(0.30, 0.75)
        time.sleep(sleep_time)

        # 2. Artificial CPU spike
        _simulate_cpu_burn(random.uniform(0.03, 0.07))

        # 3. Artificial memory spike (bounded leak + transient allocation)
        if len(_LEAK_BUFFER) < _MAX_LEAK_CHUNKS:
            _LEAK_BUFFER.append(os.urandom(1024 * 1024))  # Retain 1 MB chunk
        _transient_spike = [random.random() for _ in range(250_000)]
    else:
        # Normal fast response for stable version
        time.sleep(random.uniform(0.002, 0.015))

    latency = time.perf_counter() - start_time
    REQUEST_LATENCY.set(latency)
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

    # Measure actual process CPU & RSS memory via psutil
    proc_cpu = PROCESS.cpu_percent(interval=None)
    sys_cpu = psutil.cpu_percent(interval=None)
    cpu_val = max(proc_cpu, sys_cpu)

    mem_mb = PROCESS.memory_info().rss / (1024 * 1024)

    # Ensure realistic baseline noise or fallback readings if scraped during low traffic
    if VERSION == "canary_poisoned":
        if cpu_val < 15.0:
            cpu_val += random.uniform(35.0, 65.0)
        # If no requests have hit "/" yet, still reflect poisoned latency & memory
        if REQUEST_LATENCY._value.get() == 0.0:
            REQUEST_LATENCY.set(random.uniform(0.35, 0.75))
        if len(_LEAK_BUFFER) == 0:
            mem_mb += random.uniform(45.0, 85.0)
    else:
        if cpu_val == 0.0:
            cpu_val = random.uniform(1.5, 6.5)
        if REQUEST_LATENCY._value.get() == 0.0:
            REQUEST_LATENCY.set(random.uniform(0.003, 0.015))

    CPU_USAGE.set(round(cpu_val, 3))
    MEMORY_USAGE.set(round(mem_mb, 3))

    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
