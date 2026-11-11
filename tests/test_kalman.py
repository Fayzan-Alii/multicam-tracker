import numpy as np
import pytest

from multicam_tracker.config import MotionConfig
from multicam_tracker.tracking.kalman import GroundMotion, ImageMotion, KalmanFilter, mahalanobis_sq


def test_scalar_update_matches_hand_computation():
    kf = KalmanFilter()
    # Prior 0 with variance 4, measurement 2 with variance 4 -> gain 0.5.
    mean, cov = kf.update(np.array([0.0]), np.array([[4.0]]), np.array([2.0]), np.array([[1.0]]), np.array([[4.0]]))
    np.testing.assert_allclose(mean, [1.0])
    np.testing.assert_allclose(cov, [[2.0]])


def test_ground_predict_moves_mean_and_adds_acceleration_noise():
    model = GroundMotion(MotionConfig(space="ground", accel_noise=2.0))
    mean, cov = model.predict(np.array([0.0, 0.0, 1.0, 2.0]), np.zeros((4, 4)), dt=0.5)
    np.testing.assert_allclose(mean, [0.5, 1.0, 1.0, 2.0])
    # Discrete white-noise acceleration, sigma_a^2 = 4, dt = 0.5: [dt^4/4, dt^3/2; dt^3/2, dt^2] * 4.
    np.testing.assert_allclose(cov[0, 0], 0.0625)
    np.testing.assert_allclose(cov[0, 2], 0.25)
    np.testing.assert_allclose(cov[2, 2], 1.0)
    assert cov[0, 1] == 0.0


def test_ground_initiate_uses_measurement_covariance():
    model = GroundMotion(MotionConfig(space="ground", init_speed_std=1.5))
    mean, cov = model.initiate(np.array([3.0, 4.0]), np.diag([0.01, 0.04]))
    np.testing.assert_allclose(mean, [3.0, 4.0, 0.0, 0.0])
    np.testing.assert_allclose(np.diag(cov), [0.01, 0.04, 2.25, 2.25])


def test_ground_measurement_noise_has_a_floor():
    model = GroundMotion(MotionConfig(space="ground", min_meas_std=0.1))
    R = model.measurement_noise(np.diag([1e-6, 0.5]))
    np.testing.assert_allclose(np.diag(R), [0.01, 0.5])


def test_ground_filter_recovers_constant_velocity():
    model = GroundMotion(MotionConfig(space="ground", accel_noise=0.5))
    R = np.diag([0.01, 0.01])
    mean, cov = model.initiate(np.array([0.0, 0.0]), R)
    for k in range(1, 31):
        mean, cov = model.predict(mean, cov, dt=0.5)
        mean, cov = model.update(mean, cov, np.array([0.6 * k, -0.4 * k]), R)
    np.testing.assert_allclose(mean[2:], [1.2, -0.8], atol=1e-2)


def test_mahalanobis_uses_innovation_covariance():
    model = GroundMotion(MotionConfig(space="ground"))
    cov = np.diag([1.0, 1.0, 5.0, 5.0])
    d2 = model.gating_distance(np.zeros(4), cov, np.array([[2.0, 0.0], [0.0, 1.0]]), np.repeat(np.eye(2)[None], 2, 0))
    np.testing.assert_allclose(d2, [2.0, 0.5])


def test_mahalanobis_helper_on_batch():
    d2 = mahalanobis_sq(np.array([[3.0, 4.0]]), np.array([np.diag([9.0, 16.0])]))
    np.testing.assert_allclose(d2, [2.0])


def test_image_initiate_follows_bytetrack_parameterisation():
    model = ImageMotion(MotionConfig(space="image"), fps=30.0)
    mean, cov = model.initiate(np.array([10.0, 20.0, 30.0, 120.0]))
    np.testing.assert_allclose(mean, [20.0, 70.0, 0.2, 100.0, 0, 0, 0, 0])
    std = np.sqrt(np.diag(cov))
    np.testing.assert_allclose(std, [10.0, 10.0, 1e-2, 10.0, 6.25, 6.25, 1e-5, 6.25])


def test_image_mean_converts_back_to_box():
    model = ImageMotion(MotionConfig(space="image"), fps=30.0)
    mean, _ = model.initiate(np.array([10.0, 20.0, 30.0, 120.0]))
    np.testing.assert_allclose(model.to_box(mean), [10.0, 20.0, 30.0, 120.0])


def test_image_predict_scales_with_elapsed_frames():
    model = ImageMotion(MotionConfig(space="image"), fps=10.0)
    mean = np.array([0.0, 0.0, 0.5, 100.0, 2.0, -1.0, 0.0, 0.0])
    moved, _ = model.predict(mean, np.eye(8), dt=0.3)  # 3 frames at 10 fps
    np.testing.assert_allclose(moved[:4], [6.0, -3.0, 0.5, 100.0])


def test_update_rejects_non_positive_definite_innovation():
    kf = KalmanFilter()
    with pytest.raises(np.linalg.LinAlgError):
        kf.update(np.zeros(1), np.zeros((1, 1)), np.zeros(1), np.eye(1), np.zeros((1, 1)))
