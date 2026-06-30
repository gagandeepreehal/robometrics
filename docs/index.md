# RoboMetrics

<div class="rm-hero" markdown="1">
RoboMetrics is a lightweight Python library for evaluating robotics and autonomy outputs with typed, local metric functions. It covers trajectories, multi-modal prediction, driving safety, comfort, physics feasibility, task outcomes, manipulation signals, calibration, coverage, and experiment comparison without requiring a simulator, dashboard, ROS install, or cloud service.
</div>

<div class="rm-badges" markdown="1">
<span class="rm-badge">Local-first</span>
<span class="rm-badge">Typed Python API</span>
<span class="rm-badge">CLI-ready JSON</span>
<span class="rm-badge">Simulator-agnostic</span>
</div>

<div class="rm-grid rm-grid-3" markdown="1">
<div class="rm-card" markdown="1">
[Start in Python](quickstart.md)

Install the package, run `Evaluator`, and inspect `EvaluationResult` output.
</div>

<div class="rm-card" markdown="1">
[Start from files](guides/policy_output.md)

Validate CSV or JSON policy outputs, evaluate them, and generate a report.
</div>

<div class="rm-card" markdown="1">
[Pick metrics](metrics.md)

Browse the metric families, units, directionality, and edge-case behavior.
</div>
</div>

<div class="rm-callout" markdown="1">
Prefer a notebook? Open the
[Colab demo](https://colab.research.google.com/github/gagandeepreehal/robometrics/blob/main/notebooks/robometrics_colab_demo.ipynb)
and run the same local-first evaluation flow without cloning the repository.
</div>

## Installation

```bash
pip install robometrics
```

Install optional I/O support when you want CSV helpers that use pandas:

```bash
pip install "robometrics[io]"
```

Install only the extras you need:

```bash
pip install "robometrics[mcap]"      # MCAP JSON-message adapter
pip install "robometrics[loggers]"   # W&B and MLflow logging helpers
```

## 30-Second Evaluation

```python
import numpy as np
from robometrics import Evaluator

predictions = [
    np.array([[0.0, 0.0], [1.1, 0.0], [2.0, 0.0]]),
    np.array([[0.0, 0.0], [1.3, 0.0], [2.5, 0.0]]),
]
ground_truths = [
    np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]),
    np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]),
]

result = Evaluator().evaluate_dataset(
    predictions=predictions,
    ground_truths=ground_truths,
    metrics=["ade", "fde"],
    thresholds={"ade": 0.5, "fde": 1.0},
    bootstrap_ci=1000,
)

print(result.to_markdown())
```

## Choose Your Path

<div class="rm-grid rm-grid-2" markdown="1">
<div class="rm-card" markdown="1">
**Evaluate policy outputs**

Use the [policy-output guide](guides/policy_output.md) when your data comes from Isaac, MuJoCo, LeRobot, ROS 2, logs, or a custom rollout exporter.
</div>

<div class="rm-card" markdown="1">
**Compare checkpoints**

Use [Comparing Policies](guides/comparing_policies.md) for baseline-vs-candidate comparisons and `EvaluationResult.compare()`.
</div>

<div class="rm-card" markdown="1">
**Automate CI checks**

Use [CI Integration](guides/ci_integration.md) to wire `robometrics compare` into GitHub Actions.
</div>

<div class="rm-card" markdown="1">
**Extend the registry**

Use [Writing a Pack](guides/writing_a_pack.md) when your project needs custom metrics while keeping the same evaluator and CLI surface.
</div>
</div>

## What To Read Next

| Need | Page |
| --- | --- |
| Confirm project fit and boundaries | [What RoboMetrics Is And Isn't](what_it_is.md) |
| Learn accepted input shapes, units, and return types | [Input And Output Contract](input_output_contract.md) |
| Try the fastest local workflow | [Quickstart](quickstart.md) |
| Run multiple metrics together | [Evaluation Guide](evaluation.md) |
| Load directories or matched datasets | [Loader Examples](guides/loader_examples.md) |
| Handle result JSON versions | [Result Schema Migration](guides/result_schema_migration.md) |
| Use the shell interface | [CLI Commands](cli.md) |
| Browse metric units and direction | [Metric Catalog](metrics.md) |
| Check exported symbols | [API Reference](api.md) |

<div class="rm-callout" markdown="1">
RoboMetrics is the metrics layer. It does not replace a simulator, planner, replay system, dashboard, or experiment tracker; it gives those systems a small, explicit evaluation contract.
</div>
