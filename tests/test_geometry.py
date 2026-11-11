import numpy as np
import pytest

from multicam_tracker.geometry.camera import Camera, HomographyCamera
from multicam_tracker.geometry.homography import estimate_homography, ransac_homography
from multicam_tracker.geometry.uncertainty import foot_covariance, foot_points, propagate_to_ground

K = np.array([[100.0, 0.0, 320.0], [0.0, 100.0, 240.0], [0.0, 0.0, 1.0]])


def top_down_camera():
    # 10 m above the origin looking straight down; image x = world x, image y = -world y.
    R = np.diag([1.0, -1.0, -1.0])
    return Camera(K=K, R=R, t=np.array([0.0, 0.0, 10.0]), image_size=(480, 640))


def horizontal_camera():
    # 2 m above the origin looking along world +Y; the horizon is the row v = 240.
    R = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
    return Camera(K=K, R=R, t=np.array([0.0, 2.0, 0.0]), image_size=(480, 640))


def test_camera_centre_is_recovered_from_extrinsics():
    np.testing.assert_allclose(top_down_camera().center, [0.0, 0.0, 10.0])
    np.testing.assert_allclose(horizontal_camera().center, [0.0, 0.0, 2.0])


def test_ground_point_projects_to_hand_computed_pixel():
    uv = top_down_camera().ground_to_image(np.array([[1.0, 2.0]]))
    np.testing.assert_allclose(uv, [[330.0, 220.0]])


def test_pixel_backprojects_to_ground_point():
    xy, valid = horizontal_camera().image_to_ground(np.array([[320.0, 340.0]]))
    assert valid.tolist() == [True]
    np.testing.assert_allclose(xy, [[0.0, 2.0]], atol=1e-9)


def test_pixel_above_horizon_is_invalid():
    _, valid = horizontal_camera().image_to_ground(np.array([[320.0, 100.0], [320.0, 400.0]]))
    assert valid.tolist() == [False, True]


def test_project_marks_points_behind_camera():
    uv, in_front = horizontal_camera().project(np.array([[0.0, 5.0, 0.0], [0.0, -5.0, 0.0]]))
    assert in_front.tolist() == [True, False]
    np.testing.assert_allclose(uv[0], [320.0, 280.0])


def test_distorted_camera_undistorts_before_backprojection():
    cam = top_down_camera()
    dist = np.array([-0.2, 0.05, 0.0, 0.0, 0.0])
    distorted = Camera(K=K, R=cam.R, t=cam.t, dist=dist, image_size=cam.image_size)
    world = np.array([[2.0, -1.5], [-1.0, 1.0]])
    import cv2

    pts3 = np.hstack([world, np.zeros((2, 1))])
    rvec, _ = cv2.Rodrigues(cam.R)
    uv, _ = cv2.projectPoints(pts3, rvec, cam.t, K, dist)
    xy, valid = distorted.image_to_ground(uv.reshape(-1, 2))
    assert valid.all()
    np.testing.assert_allclose(xy, world, atol=1e-4)


def test_foot_point_is_bottom_centre_of_box():
    np.testing.assert_allclose(foot_points(np.array([[10.0, 20.0, 30.0, 100.0]])), [[20.0, 100.0]])


def test_foot_covariance_scales_with_box_and_respects_floor():
    cov = foot_covariance(np.array([[0.0, 0.0, 50.0, 200.0], [0.0, 0.0, 5.0, 10.0]]), 0.1, 0.05, 2.0)
    np.testing.assert_allclose(cov[0], np.diag([25.0, 100.0]))
    np.testing.assert_allclose(cov[1], np.diag([4.0, 4.0]))


def test_covariance_through_affine_view_is_exact():
    # For the top-down camera the map is x = (u-320)/10, y = -(v-240)/10, so J = diag(0.1, -0.1).
    xy, cov, valid = propagate_to_ground(
        top_down_camera(), np.array([[330.0, 220.0]]), np.array([np.diag([4.0, 9.0])])
    )
    assert valid.all()
    np.testing.assert_allclose(xy, [[1.0, 2.0]])
    np.testing.assert_allclose(cov[0], np.diag([0.04, 0.09]), atol=1e-12)


def test_covariance_matches_finite_difference_jacobian():
    cam = horizontal_camera()
    uv = np.array([[350.0, 300.0]])
    cov_px = np.array([[[9.0, 2.0], [2.0, 16.0]]])
    _, cov, _ = propagate_to_ground(cam, uv, cov_px)

    eps = 1e-4
    J = np.zeros((2, 2))
    for k in range(2):
        d = np.zeros(2)
        d[k] = eps
        plus, _ = cam.image_to_ground(uv + d)
        minus, _ = cam.image_to_ground(uv - d)
        J[:, k] = (plus[0] - minus[0]) / (2 * eps)
    np.testing.assert_allclose(cov[0], J @ cov_px[0] @ J.T, rtol=1e-5)


def test_uncertainty_grows_towards_the_horizon():
    cam = horizontal_camera()
    cov_px = np.repeat(np.diag([4.0, 4.0])[None], 2, axis=0)
    _, cov, _ = propagate_to_ground(cam, np.array([[320.0, 440.0], [320.0, 250.0]]), cov_px)
    near, far = np.linalg.det(cov[0]), np.linalg.det(cov[1])
    assert far > 100 * near


def test_homography_recovers_affine_map():
    H = np.array([[2.0, 0.0, 1.0], [0.0, 3.0, -2.0], [0.0, 0.0, 1.0]])
    src = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 5.0]])
    dst = np.array([[1.0, -2.0], [3.0, -2.0], [1.0, 1.0], [3.0, 1.0], [5.0, 13.0]])
    np.testing.assert_allclose(estimate_homography(src, dst), H, atol=1e-9)


def test_homography_recovers_projective_map_on_held_out_points():
    H = np.array([[1.2, 0.1, 30.0], [-0.05, 0.9, 12.0], [1e-3, 2e-3, 1.0]])
    rng = np.random.default_rng(0)
    src = rng.uniform(0, 500, size=(12, 2))

    def apply(pts):
        p = np.hstack([pts, np.ones((len(pts), 1))]) @ H.T
        return p[:, :2] / p[:, 2:]

    est = estimate_homography(src[:8], apply(src[:8]))
    held = src[8:]
    q = np.hstack([held, np.ones((len(held), 1))]) @ est.T
    np.testing.assert_allclose(q[:, :2] / q[:, 2:], apply(held), atol=1e-6)


def test_homography_needs_four_points():
    with pytest.raises(ValueError, match="4"):
        estimate_homography(np.zeros((3, 2)), np.zeros((3, 2)))


def test_ransac_rejects_outliers():
    rng = np.random.default_rng(1)
    src = rng.uniform(0, 100, size=(40, 2))
    dst = src * 2.0 + np.array([5.0, -3.0])
    outliers = np.zeros(40, dtype=bool)
    outliers[:12] = True
    dst[outliers] += rng.uniform(20, 60, size=(12, 2))

    H, inliers = ransac_homography(src, dst, threshold=1.0, iterations=500, seed=0)

    assert inliers.tolist() == (~outliers).tolist()
    np.testing.assert_allclose(H, [[2.0, 0.0, 5.0], [0.0, 2.0, -3.0], [0.0, 0.0, 1.0]], atol=1e-6)


def test_homography_camera_roundtrip_and_validity():
    cam = horizontal_camera()
    ground = np.array([[-2.0, 3.0], [2.0, 3.0], [2.0, 10.0], [-2.0, 10.0], [0.0, 6.0]])
    image = cam.ground_to_image(ground)
    hcam = HomographyCamera.from_correspondences(image, ground, image_size=(480, 640))

    xy, valid = hcam.image_to_ground(np.array([[320.0, 340.0], [320.0, 100.0]]))
    assert valid.tolist() == [True, False]
    np.testing.assert_allclose(xy[0], [0.0, 2.0], atol=1e-6)
    np.testing.assert_allclose(hcam.ground_to_image(np.array([[0.0, 2.0]])), [[320.0, 340.0]], atol=1e-6)


def test_look_at_points_the_optical_axis_at_the_target():
    from multicam_tracker.geometry.camera import look_at

    center = np.array([3.0, -15.0, 6.0])
    target = np.array([3.0, 9.0, 0.0])
    R = look_at(center, target)
    cam = Camera(K=K, R=R, t=-R @ center, image_size=(480, 640))
    uv, in_front = cam.project(target[None])
    assert in_front.all()
    np.testing.assert_allclose(uv, [[320.0, 240.0]], atol=1e-9)
    np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-12)
    # World "up" must appear as image "up" (decreasing v).
    above, _ = cam.project((target + [0.0, 0.0, 1.0])[None])
    assert above[0, 1] < 240.0
