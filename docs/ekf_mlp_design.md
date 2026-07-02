# EKF–MLP Hybrid Estimator Design

## Motivation

A self-balancing robot is an unstable inverted-pendulum-like system. Stable control requires accurate state estimation.

A conventional EKF can fuse IMU, encoder, and vision measurements, but its performance depends on the accuracy of:

- The system model
- The measurement model
- Noise covariance assumptions
- Sensor bias modeling
- Wheel-ground contact assumptions

Under slip, slope, disturbance, or sensor bias, model-based estimation error can increase.

## Main Idea

The proposed estimator combines:

- EKF: model-based state estimation
- MLP: learning-based residual/error compensation

The MLP is not intended to replace the EKF. Instead, it compensates systematic residual errors that appear when the EKF model is imperfect.

## Candidate Architectures

### Option A: Measurement Correction

The MLP predicts sensor measurement error.

```text
raw sensor measurement -> MLP correction -> corrected measurement -> EKF update
```

Example:

```text
z_corrected = z_measured + Δz_MLP
```

This is useful when encoder, gyro, or vision measurements have systematic errors.

### Option B: State Residual Correction

The MLP predicts EKF state estimation error.

```text
sensor measurements -> EKF -> x_EKF
sensor/history features -> MLP -> Δx
x_corrected = x_EKF + Δx
```

This is useful when the EKF output itself has systematic bias.

### Option C: Adaptive Noise Tuning

The MLP predicts noise covariance scaling.

```text
sensor/history features -> MLP -> Q/R scaling
EKF uses adaptive Q/R
```

This is more advanced and should be considered after Options A and B.

## Recommended First Implementation

Start with Option A.

Reason:

- The current dataset collector already generates target measurement residuals.
- Measurement correction is easier to validate.
- It is less likely to destabilize the full estimator than direct state correction.

## Candidate MLP Input

Possible input vector:

```text
[
  gyro_x, gyro_y, gyro_z,
  acc_x, acc_y, acc_z,
  enc_l_vel, enc_r_vel,
  enc_forward_vel,
  vision_x, vision_y, vision_yaw,
  speed_setpoint,
  yaw_setpoint,
  lqr_motor_cmd_l,
  lqr_motor_cmd_r
]
```

A later version can include time-history windows.

## Candidate MLP Output

Initial output candidates:

```text
[
  encoder_forward_vel_error,
  gyro_x_error,
  vision_x_error,
  vision_y_error,
  vision_yaw_error
]
```

## Evaluation Metrics

Estimator performance should be compared using:

- RMSE of pitch estimate
- RMSE of forward velocity estimate
- RMSE of yaw estimate
- Position estimation error
- Fall rate under disturbance
- Control success rate
- Recovery time after disturbance

## Baselines

Compare at least three cases:

- Raw sensor measurement
- EKF-only estimation
- EKF + MLP correction

## Environment Cases

Test under:

- Flat ground
- Low-friction/slip condition
- Slope condition
- External impulse disturbance
- Sensor bias condition
- Combined uncertainty condition
