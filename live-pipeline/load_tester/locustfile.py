import os
from locust import HttpUser, between, task

CANARY_HOST = os.getenv("CANARY_HOST", "").rstrip("/")


class TrafficUser(HttpUser):
    """
    Simulates continuous user traffic to the API's '/' endpoint.
    If CANARY_HOST is configured, routes ~20% of requests to the canary instance
    to simulate a real-world 80/20 canary deployment traffic split.
    """

    wait_time = between(0.2, 1.0)

    @task(4)
    def hit_root_endpoint(self):
        self.client.get("/", name="/ (stable)")

    @task(1)
    def hit_canary_endpoint(self):
        if CANARY_HOST:
            self.client.get(f"{CANARY_HOST}/", name="/ (canary)")
        else:
            self.client.get("/", name="/ (stable)")
