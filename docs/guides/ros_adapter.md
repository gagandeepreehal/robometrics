# ROS Adapter

The `robometrics.ros` module converts common ROS message shapes into NumPy arrays that can be passed to the same metrics used elsewhere in RoboMetrics. The adapter does not import ROS packages at module import time. Each function checks the relevant ROS package only when called and raises a clear `ImportError` if a real ROS message is used without ROS installed and sourced. For tests and simple scripts, the functions also accept plain Python objects that duck-type the same attribute paths, such as `types.SimpleNamespace`.

Install ROS 2 Humble or later and source your workspace before using real ROS messages. The adapter covers `nav_msgs/Path`, sequences of `nav_msgs/Odometry`, `geometry_msgs/PoseArray`, and `visualization_msgs/MarkerArray`. Paths and odometry sequences become `Nx3` arrays. Marker arrays are grouped by namespace, sorted by marker id inside each namespace, and returned as a list of actor trajectories.

```python
from robometrics import min_distance_to_actors
from robometrics.ros import from_odometry_sequence, from_path_msg

# In a ROS node, these would be real message objects.
ego_traj = from_path_msg(path_msg)
actor_traj = from_odometry_sequence(actor_odometry_msgs)

distance = min_distance_to_actors(
    ego_traj[:, :2],
    [actor_traj[:, :2]],
)
print(distance)
```

You can run the same conversion logic in tests without ROS:

```python
from types import SimpleNamespace
from robometrics.ros import from_path_msg

def position(x, y, z):
    return SimpleNamespace(x=x, y=y, z=z)

msg = SimpleNamespace(
    poses=[
        SimpleNamespace(pose=SimpleNamespace(position=position(0.0, 0.0, 0.0))),
        SimpleNamespace(pose=SimpleNamespace(position=position(1.0, 0.0, 0.0))),
    ]
)
traj = from_path_msg(msg)
print(traj.shape)
```

For safety suites, convert ego and actor trajectories first, then call `collision_rate`, `collision_rate_obb`, `min_distance_to_actors`, or `time_to_collision` depending on the information available. Use OBB collision when dimensions and yaw are available; use disc collision when you only have positions and conservative radii.
