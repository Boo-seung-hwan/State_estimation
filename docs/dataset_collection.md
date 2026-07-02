# LQR Dataset Collection

## Purpose

The dataset collector converts the headless LQR-controlled MuJoCo simulation into a research dataset for state estimation.

The dataset is intended for:

- EKF-only state estimation
- MLP-based residual/error estimation
- EKF–MLP hybrid estimator training
- Comparison between ground truth, noisy sensor measurements, and corrected estimates

## Data Source

The dataset is generated from the MuJoCo self-balancing robot simulation.

The robot is stabilized by the LQR baseline controller. During each episode, the script records:

- MuJoCo ground-truth state
- LQR control information
- Simulated IMU measurements
- Simulated encoder measurements
- Simulated vision/position measurements
- Supervised residual targets for learning-based correction

## Example Command

```bash
python scripts/collect_lqr_dataset.py \
  --model src/simulation/scene.xml \
  --out datasets/lqr_dataset.csv \
  --episodes 50 \
  --duration 10 \
  --control-hz 200 \
  --cmd-mode random_step \
  --max-speed 0.1 \
  --max-yaw 0.6 \
  --command-interval 2.0 \
  --seed 0
```

## Command Modes

### `zero`

The robot balances in place.

Useful for:

- Checking basic LQR stability
- Collecting near-equilibrium data
- Verifying sensor noise and estimator behavior around the upright state

### `random_step`

The speed and yaw commands change in a stepwise random manner.

Useful for:

- Collecting richer dynamic trajectories
- Training an MLP with more diverse motion
- Testing estimator robustness under command changes

### `sine`

The speed and yaw commands vary sinusoidally.

Useful for:

- Smooth trajectory testing
- Frequency-like response analysis
- Checking estimator behavior under continuous motion

## Dataset Policy

Large generated CSV files should not be committed directly to Git.

Recommended policy:

- Keep full datasets outside Git.
- Commit only `datasets/README.md`.
- Commit small sample files with the name pattern `sample_*.csv`.
- Use external storage or Git LFS later if large datasets must be versioned.
