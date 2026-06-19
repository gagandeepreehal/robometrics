# Performance

RoboMetrics keeps safety metrics NumPy-first so they remain practical in local
CI. `lane_departure_rate()` and `offroad_rate()` use vectorized polygon
containment through `points_in_polygon()`. `collision_rate_obb()` evaluates all
timesteps for each actor with a batched separating-axis test instead of a Python
loop per timestep.

Use this smoke benchmark when changing geometry kernels:

```python
import time
import numpy as np

from robometrics import collision_rate_obb, lane_departure_rate

n = 10_000
ego = np.column_stack([np.linspace(0.0, 100.0, n), np.zeros(n)])
lane = np.array([[-1.0, -5.0], [101.0, -5.0], [101.0, 5.0], [-1.0, 5.0]])

start = time.perf_counter()
lane_departure_rate(ego, lane)
print("lane_departure_rate", time.perf_counter() - start)

actors = [ego + np.array([0.2, 0.0]) for _ in range(10)]
yaws = np.zeros(n)
dims = np.array([4.5, 2.0])

start = time.perf_counter()
collision_rate_obb(
    ego_traj=ego,
    ego_dims=dims,
    ego_yaws=yaws,
    actor_trajs=actors,
    actor_dims=[dims for _ in actors],
    actor_yaws=[yaws for _ in actors],
)
print("collision_rate_obb", time.perf_counter() - start)
```

The launch target is that 10 Hz, 10-second scenario checks with 10 actors stay
comfortably below one second on a typical developer laptop.
