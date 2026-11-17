import math
import os
from locust import HttpUser, LoadTestShape, between, task

CANARY_HOST = os.getenv("CANARY_HOST", "").rstrip("/")


class TrafficUser(HttpUser):
    """
    Simulates continuous user traffic to the API's '/' endpoint.
    Routes ~80% of requests to the stable instance and ~20% to the canary instance.
    """

    wait_time = between(0.2, 0.8)

    @task(4)
    def hit_root_endpoint(self):
        self.client.get("/", name="/ (stable)")

    @task(1)
    def hit_canary_endpoint(self):
        if CANARY_HOST:
            self.client.get(f"{CANARY_HOST}/", name="/ (canary)")
        else:
            self.client.get("/", name="/ (stable)")


class RealisticTrafficShape(LoadTestShape):
    """
    Simulates non-stationary, real-world server traffic with repeating 60-second cycles:
      - 00s - 20s : Normal fluctuating diurnal load (18 - 45 users)
      - 20s - 35s : Sudden Flash-Crowd User Influx / Traffic Surge (110 users!)
      - 35s - 50s : Post-surge moderate load (30 users)
      - 50s - 60s : Quiet off-peak trough (8 users)
    """

    cycle_length = 60.0

    def tick(self):
        run_time = self.get_run_time()
        phase = run_time % self.cycle_length

        if phase < 20.0:
            # Smooth wave between 18 and 45 users
            users = int(31 + 14 * math.sin((phase / 20.0) * math.pi))
            return (users, 25)
        elif phase < 35.0:
            # Sudden influx of users (flash crowd spike)
            return (110, 60)
        elif phase < 50.0:
            # Moderate load
            return (30, 40)
        else:
            # Quiet trough
            return (8, 30)
