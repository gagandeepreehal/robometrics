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
| ROS 2 bag JSON | `ros2`, `ros2-json`, `ros2-bag-json` | none; expects exported JSON |
| LeRobot-style JSON | `lerobot`, `lerobot-style` | none; expects small JSON exports |
| RLDS-style JSON | `rlds`, `rlds-style` | none; expects small JSON exports |
| MCAP JSON messages | `mcap` | optional `robometrics[mcap]` extra |

Generic CSV files load `x` and `y` columns and preserve an optional `z` column
as a third coordinate. Validation reports the same 2D or 3D dimensionality that
`load(path)` returns in the `Trajectory` schema.

The ROS, ROS 2, LeRobot, RLDS, and MCAP adapters intentionally do not import
ROS, `rclpy`, TensorFlow, LeRobot, or MCAP at package import time. Export heavy
datasets to small CSV/JSON fixtures for CI gates, and keep full dataset loaders
in project code when they require large dependencies.

## ROS 2 Bag JSON Export

Use `get_adapter("ros2-json")` for JSON exported from ROS 2 bag tooling. The
adapter does not open `.db3` bag storage and does not import `rclpy`; it expects
a JSON file with a `messages`, `records`, or `data` list. Each record may use
`message`, `msg`, or `data` for the decoded message body.

```json
{
  "messages": [
    {
      "topic": "/odom",
      "timestamp": 1.0,
      "message": {
        "pose": {
          "pose": {
            "position": {"x": 0.0, "y": 0.0, "z": 0.0}
          }
        }
      }
    }
  ]
}
```

```python
from robometrics.adapters import get_adapter

adapter = get_adapter("ros2-json")
trajectory = adapter.load("ros2_messages.json")
```

## MCAP

Install the optional extra before reading MCAP files:

```bash
pip install "robometrics[mcap]"
```

The built-in MCAP adapter reads messages whose payload is UTF-8 JSON shaped
like generic RoboMetrics trajectory JSON or ROS-style exported JSON. Binary ROS
2 CDR decoding is deliberately left to project-specific export steps so the
core package stays lightweight.

## Adding An Adapter

Create a class with `load`, `validate`, and `metadata` methods. Keep the core
import path free of heavyweight imports. If a format needs an optional package,
raise an `ImportError` that names the missing dependency and the install path.
Add tiny synthetic fixtures in tests rather than committing full datasets.
