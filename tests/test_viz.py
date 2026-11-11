import numpy as np
import pytest
from PIL import Image

from multicam_tracker.viz.bev import BEVCanvas, covariance_ellipse
from multicam_tracker.viz.colors import id_color
from multicam_tracker.viz.composite import compose, save_gif


def test_id_colors_are_deterministic_and_distinct():
    assert id_color(3) == id_color(3)
    colors = {id_color(i) for i in range(1, 30)}
    assert len(colors) == 29


def test_covariance_ellipse_axes_are_two_sigma():
    axes, angle = covariance_ellipse(np.diag([4.0, 1.0]))
    np.testing.assert_allclose(axes, [4.0, 2.0])
    assert angle == pytest.approx(0.0, abs=1e-9)
    c, s = np.cos(np.pi / 4), np.sin(np.pi / 4)
    R = np.array([[c, -s], [s, c]])
    axes, angle = covariance_ellipse(R @ np.diag([9.0, 1.0]) @ R.T)
    np.testing.assert_allclose(axes, [6.0, 2.0])
    assert angle % 180 == pytest.approx(45.0)


def test_bev_is_a_true_top_down_view():
    # Seen from above with x to the right, +y must point up the canvas (not a mirror image).
    bev = BEVCanvas(roi=(0.0, 0.0, 10.0, 5.0), px_per_m=10)
    assert bev.size == (50, 100)  # H, W
    np.testing.assert_allclose(bev.world_to_px(np.array([[0.0, 0.0], [10.0, 5.0]])), [[0, 50], [100, 0]])


def test_tall_roi_is_rotated_to_landscape_not_mirrored():
    # Rotating the top view 90 degrees clockwise: +y points right and +x points down.
    bev = BEVCanvas(roi=(-3.0, -9.0, 9.0, 27.0), px_per_m=10)
    assert bev.size == (120, 360)
    np.testing.assert_allclose(bev.world_to_px(np.array([[-3.0, 27.0], [9.0, -9.0]])), [[360, 0], [0, 120]])


def test_drawn_track_colours_its_ground_position():
    bev = BEVCanvas(roi=(0.0, 0.0, 10.0, 10.0), px_per_m=20)
    img = bev.render(tracks=[(7, np.array([5.0, 5.0]), np.eye(2) * 0.01)])
    u, v = bev.world_to_px(np.array([[5.0, 5.0]]))[0].astype(int)
    assert tuple(int(c) for c in img[v, u]) == id_color(7)


def test_compose_tiles_panels_next_to_the_map():
    panels = [np.zeros((270, 480, 3), np.uint8)] * 3
    out = compose(panels, np.zeros((100, 300, 3), np.uint8), panel_width=480, cols=2)
    assert out.shape == (540, 960 + 540 * 3, 3)


def test_save_gif_writes_all_frames(tmp_path):
    frames = [np.full((20, 40, 3), 30 * i, np.uint8) for i in range(4)]
    path = tmp_path / "demo.gif"
    save_gif(frames, path, fps=2, width=20)
    with Image.open(path) as gif:
        assert gif.n_frames == 4 and gif.size == (20, 10)
