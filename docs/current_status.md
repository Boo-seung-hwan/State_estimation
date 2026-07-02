# Current Status

## Project Goal

The goal of this project is to develop a hybrid state estimation framework for a two-wheeled self-balancing robot under model uncertainty.

The current research direction is:

- Use MuJoCo as the main simulation environment.
- Stabilize the robot with an LQR baseline controller.
- Collect ground-truth and pseudo-sensor datasets from the LQR-controlled robot.
- Implement an EKF-based state estimator.
- Train an MLP to compensate EKF estimation residuals.
- Compare EKF-only and EKF–MLP hybrid estimation under slip, slope, and disturbance conditions.

## Current Implementation Status

### Completed

- MuJoCo XML model for a two-wheeled self-balancing robot.
- Basic scene with floor contact and wheel-ground friction.
- LQR-based baseline balance controller.
- GUI simulation script.
- Headless LQR simulation script.
- LQR dataset collector script.
- Docker-based development environment.

### Verified

- MuJoCo model loading works.
- Headless physics stepping works.
- LQR baseline can stabilize the robot under basic conditions.
- Dataset collection script can save CSV logs from LQR-controlled trajectories.

### Known Issues

- GUI rendering can fail in some Docker + WSLg + Qt/OpenGL environments.
- Headless scripts should be prioritized for research experiments.
- Current sensor data are simulated pseudo-measurements, not real hardware measurements.

## Next Development Stage

The next stage is to move from baseline control and dataset collection to estimator implementation.

Priority:

1. Define dataset schema.
2. Implement EKF-only estimator.
3. Train MLP residual estimator.
4. Add slip, slope, and disturbance scenarios.
5. Compare estimation and control performance.
