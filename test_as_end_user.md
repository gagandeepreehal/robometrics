# End-User Test Prompt for RoboMetrics

Paste this prompt into a fresh Claude Code session (in any directory) to test RoboMetrics as a first-time end user.

---

## Prompt

You are a new robotics engineer who just heard about the `robometrics` Python package. Your goal is to explore, install, and exercise it end-to-end — the way a real user would on day one. Work through the steps below in order. Report what you observe at each step (output, errors, surprises). Do not skip a step even if an earlier one fails; note the failure and continue.

### Step 1 — Discover the package

```bash
pip show robometrics 2>/dev/null || echo "not installed"
python -m robometrics --help 2>/dev/null || echo "CLI not available"
```

If not installed, install it:

```bash
pip install "robometrics[io]"
```

Then confirm the CLI works:

```bash
python -m robometrics --help
python -m robometrics version
```

### Step 2 — Explore available metrics

```bash
python -m robometrics list-metrics
python -m robometrics list-metrics --format json
python -m robometrics describe ade
python -m robometrics describe min_ade
python -m robometrics describe collision_rate
```

Verify that:
- The metric list is non-empty and grouped by category.
- `describe ade` returns a unit, category, and a short description.
- The JSON format is valid JSON (pipe to `python -m json.tool` to check).

### Step 3 — Python API: core metrics

Run the following script inline (use the Bash tool with a heredoc, or write it to a temp file and run it):

```python
import numpy as np
from robometrics import ade, fde, path_length, min_ade, min_fde, miss_rate

pred = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
gt   = np.array([[0.0, 0.0], [1.1, 0.0], [2.2, 0.0]])

print("ADE:", ade(pred, gt))
print("FDE:", fde(pred, gt))
print("Path length:", path_length(gt))

# Multimodal prediction
preds_k = np.array([
    [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]],
    [[0.0, 0.0], [1.1, 0.0], [2.1, 0.0]],
])
print("minADE:", min_ade(preds_k, gt))
print("minFDE:", min_fde(preds_k, gt))
print("miss_rate:", miss_rate(preds_k, gt, threshold=0.5))
```

Expected: all values are finite floats; ADE > 0; minADE <= ADE.

### Step 4 — Safety metrics

```python
import numpy as np
from robometrics import AgentState, collision_rate, min_distance_to_actors, time_to_collision

ego    = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
actors = [np.array([[0.0, 2.0], [1.0, 1.0], [2.0, 0.4]])]

print("Min dist to actors:", min_distance_to_actors(ego, actors))
print("Collision rate:", collision_rate(ego, actors, ego_radius=0.3, actor_radius=0.3))

ego_state   = AgentState(x=0.0, y=0.0, vx=2.0, vy=0.0, radius=0.3)
actor_state = AgentState(x=10.0, y=0.0, vx=0.0, vy=0.0, radius=0.3)
print("TTC:", time_to_collision(ego_state, actor_state))
```

### Step 5 — Comfort and physics metrics

```python
import numpy as np
from robometrics import acceleration, jerk, jerk_cost, smoothness_score
from robometrics import acceleration_limits_violated, dynamic_feasibility_score

traj = np.array([[0.0, 0.0], [1.0, 0.1], [2.0, 0.4], [3.0, 0.9]])
dt   = 0.5

print("Acceleration profile:", acceleration(traj, dt=dt))
print("Jerk profile:", jerk(traj, dt=dt))
print("Jerk cost:", jerk_cost(traj, dt=dt))
print("Smoothness score:", smoothness_score(traj))

print("Accel limits violated:", acceleration_limits_violated(traj, dt=dt, max_accel=5.0))
print("Dynamic feasibility:", dynamic_feasibility_score(traj, dt=dt, constraints={"max_accel": 5.0}))
```

### Step 6 — Evaluator with thresholds

```python
import numpy as np
from robometrics import Evaluator, EvaluationResult

pred = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
gt   = np.array([[0.0, 0.0], [1.1, 0.0], [2.2, 0.0]])

result = Evaluator().evaluate(
    prediction=pred,
    ground_truth=gt,
    metrics=["ade", "fde"],
    thresholds={"ade": 0.5, "fde": 1.0},
)

print(result.summary())
print(result.to_markdown())
print("Strict passed:", result.strict_passed)

# Round-trip JSON serialization
json_str   = result.to_json()
reloaded   = EvaluationResult.from_json(json_str)
print("Round-trip OK:", reloaded.summary()["metric_count"] == result.summary()["metric_count"])
```

Verify: `strict_passed` is a bool; `metric_count` matches between the original and reloaded result.

### Step 7 — CLI: validate and evaluate local fixture files

Create tiny CSV files in your current working directory:

```bash
cat > predictions.csv <<'CSV'
t,x,y
0,0.0,0.0
1,1.0,0.0
2,2.0,0.0
CSV

cat > ground_truth.csv <<'CSV'
t,x,y
0,0.0,0.0
1,1.1,0.0
2,2.1,0.0
CSV
```

Then run:

```bash
python -m robometrics validate predictions.csv
python -m robometrics validate ground_truth.csv

python -m robometrics evaluate \
  --pred predictions.csv \
  --gt   ground_truth.csv \
  --metrics ade fde \
  --threshold ade=0.5 \
  --threshold fde=1.0 \
  --output /tmp/rm_result.json

cat /tmp/rm_result.json | python -m json.tool | head -40
```

Verify: `validate` exits 0 with no errors; `evaluate` writes valid JSON containing `metrics`, `timestamp`, and `passed` fields.

### Step 8 — HTML report

```bash
python -m robometrics report /tmp/rm_result.json --output /tmp/rm_report.html
wc -l /tmp/rm_report.html
grep -c "pass\|fail" /tmp/rm_report.html
```

Verify: the HTML file is non-empty and contains pass/fail indicators.

### Step 9 — Benchmark profiles

```bash
python -m robometrics benchmark list

python -m robometrics benchmark run policy_regression_ci \
  --pred predictions.csv \
  --gt   ground_truth.csv \
  --output /tmp/rm_benchmark.json

python -m json.tool /tmp/rm_benchmark.json | head -20
```

Verify: `benchmark list` prints at least the four built-in profiles; `benchmark run` produces a JSON result with metric values.

### Step 10 — CSV IO and adapter

```python
from robometrics import load_trajectory_csv, load_trajectory_json
from robometrics.adapters import GenericCSVAdapter

csv_traj = load_trajectory_csv("predictions.csv")
print("Loaded CSV shape:", csv_traj.shape)

adapter = GenericCSVAdapter()
print("Adapter validate:", adapter.validate("predictions.csv"))
print("Adapter metadata:", adapter.metadata("predictions.csv"))
```

### Step 11 — Error handling (negative tests)

Confirm that the library raises clear errors for invalid inputs:

```python
import numpy as np
from robometrics import ade

# Shape mismatch should raise ValueError
try:
    ade(np.zeros((3, 2)), np.zeros((4, 2)))
    print("ERROR: expected ValueError not raised")
except ValueError as e:
    print("Shape mismatch caught OK:", e)

# NaN input should raise ValueError
try:
    ade(np.full((3, 2), float("nan")), np.zeros((3, 2)))
    print("ERROR: expected ValueError not raised")
except ValueError as e:
    print("NaN input caught OK:", e)
```

### Step 12 — Run the packaged examples

If the source repo is available:

```bash
python examples/basic_metrics.py
python examples/evaluator_quickstart.py
python examples/thresholds.py
python examples/load_from_csv.py
```

Each should exit 0 with no tracebacks.

---

## What to Report

After each step, note:

1. **Pass / Fail** — did the step succeed as described?
2. **Actual output** — paste the key lines (metric values, summary dicts, exit codes).
3. **Unexpected behavior** — wrong values, confusing error messages, missing features, broken CLI flags.
4. **Friction** — anything that would block or slow down a real first-time user (unclear docs, missing fixtures, confusing defaults).

At the end, write a short (10-line max) summary of:
- Which steps passed cleanly.
- Which steps had issues, and what they were.
- One or two suggestions for improving the new-user experience.
