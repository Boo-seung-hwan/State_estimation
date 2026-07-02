# Datasets

This directory is reserved for dataset descriptions and small sample datasets.

## Policy

Large generated datasets are not committed to Git.

The repository `.gitignore` should ignore:

- `datasets/*.csv`
- `datasets/*.npz`
- `datasets/*.npy`

Small sample files may be committed using the following name pattern:

```text
sample_*.csv
```

## Dataset Generation

Example:

```bash
python scripts/collect_lqr_dataset.py \
  --model src/simulation/scene.xml \
  --out datasets/lqr_dataset.csv \
  --episodes 50 \
  --duration 10 \
  --control-hz 200 \
  --cmd-mode random_step
```

## Recommended Dataset Versions

| Dataset | Purpose |
|---|---|
| `lqr_dataset_zero.csv` | Near-equilibrium balancing data |
| `lqr_dataset_step_v001.csv` | Random step command with small yaw |
| `lqr_dataset_step_v002.csv` | Random step command with larger yaw |
| `lqr_dataset_slip.csv` | Low-friction/slip condition |
| `lqr_dataset_slope.csv` | Slope condition |
| `lqr_dataset_disturbance.csv` | External disturbance condition |

## Sample Data

Only small samples should be committed.

Example:

```bash
head -n 101 datasets/lqr_dataset_step_v002.csv > datasets/sample_lqr_dataset_step_v002.csv
git add datasets/sample_lqr_dataset_step_v002.csv
```
