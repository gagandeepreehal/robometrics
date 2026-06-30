# What RoboMetrics Is And Isn't

RoboMetrics is a local metrics library for robotics evaluation. It is designed
to sit after simulation, data collection, or policy rollout code and turn
trajectory-like outputs into repeatable metric results.

## Scope Boundary

| Need | RoboMetrics role | Use another tool for |
| --- | --- | --- |
| Compute ADE, FDE, safety, comfort, coverage, calibration, physics, and task metrics | Yes | Defining the physical world or sensor stack |
| Evaluate CSV, JSON, NumPy, or lightweight adapter exports | Yes | Storing large datasets or streaming bag files directly |
| Run a deterministic CI regression check | Yes | Running distributed benchmarks or public leaderboard governance |
| Compare two local policy result files | Yes | Hosting dashboards or experiment databases |
| Parse small ROS/ROS 2 JSON exports without ROS imports | Yes | Decoding arbitrary ROS middleware state or running ROS nodes |
| Simulate robot dynamics, contacts, or scenes | No | Isaac Sim, MuJoCo, Gazebo, Drake, PyBullet, or project simulators |
| Train policies or optimize controllers | No | PyTorch, JAX, RL libraries, MPC stacks, or imitation-learning frameworks |
| Manage dataset lineage and labeling | No | Dataset/versioning platforms and lab data infrastructure |

## Use RoboMetrics When

- You already have policy predictions, rollout trajectories, reference paths,
  actor trajectories, or time-series arrays.
- You need typed metric functions that are small enough to run in unit tests or
  CI.
- You want strict JSON output for downstream CI consumers.
- You want project-local thresholds without adopting a full benchmark platform.

## Do Not Use It As

- A simulator.
- A robotics middleware.
- A training framework.
- A dataset host.
- A public leaderboard service.
- A substitute for domain-specific validation of sensor, contact, or actuation
  models.

The clean boundary is: generate or export rollouts elsewhere, then use
RoboMetrics to measure them.
