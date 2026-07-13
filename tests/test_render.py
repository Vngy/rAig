import numpy as np

from raig.core.params import make_param_frame, rest_frame
from raig.render.renderer import CHROMA_GREEN, render_to_image


def red_pixels(img):
    r, g, b, a = img[..., 0], img[..., 1], img[..., 2], img[..., 3]
    return (r > 180) & (g < 60) & (b < 60) & (a > 128)


def test_rest_pose_renders_avatar(gl_ctx, mini_rig):
    img = render_to_image(mini_rig, rest_frame())
    assert img.shape == (1024, 1024, 4)
    assert (img[..., 3] > 0).sum() > 50_000  # avatar covers a real area
    # top-left corner is empty canvas
    assert img[0, 0, 3] == 0


def test_irises_visible_at_rest(gl_ctx, mini_rig):
    img = render_to_image(mini_rig, rest_frame())
    assert red_pixels(img).sum() > 100


def test_blink_hides_irises(gl_ctx, mini_rig):
    open_img = render_to_image(mini_rig, rest_frame())
    closed = make_param_frame({"eye_l_open": 0.0, "eye_r_open": 0.0})
    closed_img = render_to_image(mini_rig, closed)
    assert red_pixels(closed_img).sum() < 0.3 * red_pixels(open_img).sum()


def test_head_turn_shifts_head_pixels(gl_ctx, mini_rig):
    rest_img = render_to_image(mini_rig, rest_frame())
    turned_img = render_to_image(mini_rig, make_param_frame({"head_angle_x": 30.0}))
    rest_cx = np.nonzero(red_pixels(rest_img))[1].mean()
    turned_cx = np.nonzero(red_pixels(turned_img))[1].mean()
    assert turned_cx - rest_cx > 10.0


def test_chroma_background(gl_ctx, mini_rig):
    img = render_to_image(mini_rig, rest_frame())
    img_green = render_to_image_bg(mini_rig)
    assert tuple(img_green[0, 0]) == (0, 255, 0, 255)
    assert img[0, 0, 3] == 0


def render_to_image_bg(rig):
    from raig.render.renderer import render_to_image

    return render_to_image(rig, rest_frame(), background=CHROMA_GREEN)
