# Adapter Guide

Adapters keep external dataset loading lightweight. Every built-in adapter uses
the same small interface:

```python
from robometrics.adapters import get_adapter

adapter = get_adapter("generic-json")
trajectory = adapter.load("trajectory.json")
report = adapter.validate("trajectory.json")
metadata = adapter.metadata("trajectory.json")
```

`load(path)` returns the standard `Trajectory` schema. `validate(path)` returns
a `DatasetValidationResult`. `metadata(path)` returns path and adapter metadata.

## Built-In Adapters

| Adapter | Name | Hard dependency |
| --- | --- | --- |
| Generic CSV | `csv`, `generic-csv` | pandas via existing CSV IO |
| Generic JSON | `json`, `generic-json` | none beyond core |
| ROS-style JSON | `ros`, `ros-style` | none; expects exported JSON |
| LeRobot-style JSON | `lerobot`, `lerobot-style` | none; expects small JSON exports |
| RLDS-style JSON | `rlds`, `rlds-style` | none; expects small JSON exports |
| MCAP placeholder | `mcap` | raises a clear optional dependency error |

The ROS, LeRobot, RLDS, and MCAP adapters intentionally do not import ROS,
TensorFlow, LeRobot, or MCAP at package import time. Export heavy datasets to
small CSV/JSON fixtures for CI gates, and keep full dataset loaders in project
code when they require large dependencies.

## Adding An Adapter

Create a class with `load`, `validate`, and `metadata` methods. Keep the core
import path free of heavyweight imports. If a format needs an optional package,
raise an `ImportError` that names the missing dependency and the install path.
Add tiny synthetic fixtures in tests rather than committing full datasets.
