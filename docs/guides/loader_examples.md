# Loader Examples

RoboMetrics loaders are small helpers for turning local trajectory files into
arrays that can be passed to metrics or `Evaluator.evaluate_dataset()`.

## Directory-Level Loading

Use `load_trajectory_dir()` when you want every supported trajectory file in a
directory as a filename-keyed dictionary:

```python
from robometrics import load_trajectory_dir

trajectories = load_trajectory_dir("rollouts/predictions")
for name, trajectory in trajectories.items():
    print(name, trajectory.shape)
```

The helper loads `.npy`, `.npz`, `.csv`, and `.json` files and skips unsupported
extensions. CSV loading requires `robometrics[io]` because it uses pandas.

## Dataset-Level Evaluation

Use `trajectories_to_dataset()` when you have matched prediction and reference
files. It preserves file order, so sort the paths or use your own manifest when
the pairing matters.

```python
from pathlib import Path

from robometrics import Evaluator
from robometrics.loaders import trajectories_to_dataset

pred_dir = Path("rollouts/predictions")
gt_dir = Path("rollouts/references")

predictions, ground_truths = trajectories_to_dataset(
    sorted(pred_dir.glob("*.csv")),
    sorted(gt_dir.glob("*.csv")),
)

result = Evaluator().evaluate_dataset(
    predictions=predictions,
    ground_truths=ground_truths,
    metrics=["ade", "fde"],
    thresholds={"ade": 0.25, "fde": 0.5},
)

print(result.to_markdown())
```

For a complete runnable example that creates temporary CSV/JSON files, loads a
directory, evaluates a matched dataset, and prints the schema version, see
`examples/dataset_loader_evaluation.py`.
