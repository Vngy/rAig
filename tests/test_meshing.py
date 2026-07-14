import numpy as np
import pytest
from PIL import Image, ImageDraw

from raig.compiler.errors import CompileError
from raig.compiler.meshing import build_mesh, grid_step_for_slot


def shape_rgba(size, draw_fn):
    img = Image.new("RGBA", size, (0, 0, 0, 0))
    draw_fn(ImageDraw.Draw(img))
    return np.array(img)


def ellipse_rgba(w=200, h=120):
    return shape_rgba((w, h), lambda d: d.ellipse((0, 0, w - 1, h - 1), fill=(255, 0, 0, 255)))


def l_shape_rgba():
    def draw(d):
        d.rectangle((0, 0, 59, 199), fill=(0, 255, 0, 255))
        d.rectangle((0, 140, 199, 199), fill=(0, 255, 0, 255))
    return shape_rgba((200, 200), draw)


def tri_areas(vertices, triangles):
    a = vertices[triangles[:, 0]]
    b = vertices[triangles[:, 1]]
    c = vertices[triangles[:, 2]]
    ab = b - a
    ac = c - a
    return 0.5 * np.abs(ab[:, 0] * ac[:, 1] - ab[:, 1] * ac[:, 0])


def test_ellipse_mesh_invariants():
    rgba = ellipse_rgba()
    mesh = build_mesh(rgba, grid_step=24)
    assert mesh.triangles.shape[0] > 4
    areas = tri_areas(mesh.vertices, mesh.triangles)
    assert (areas > 0.5).all(), "degenerate triangles present"
    assert mesh.uvs.min() >= 0.0 and mesh.uvs.max() <= 1.0
    mask_area = (rgba[..., 3] > 8).sum()
    assert areas.sum() >= 0.85 * mask_area


def test_concave_shape_centroids_inside_mask():
    rgba = l_shape_rgba()
    mask = rgba[..., 3] > 8
    mesh = build_mesh(rgba, grid_step=20)
    centroids = mesh.vertices[mesh.triangles].mean(axis=1)
    ix = np.clip(np.round(centroids[:, 0]).astype(int), 0, mask.shape[1] - 1)
    iy = np.clip(np.round(centroids[:, 1]).astype(int), 0, mask.shape[0] - 1)
    assert mask[iy, ix].all(), "triangle centroid escaped the alpha mask"
    assert tri_areas(mesh.vertices, mesh.triangles).sum() >= 0.75 * mask.sum()


def test_transparent_layer_raises():
    rgba = np.zeros((32, 32, 4), dtype=np.uint8)
    with pytest.raises(CompileError):
        build_mesh(rgba)


def test_collinear_sliver_raises_compile_error_not_qhullerror():
    # A 1px-wide vertical sliver: with grid_step=1 every sampled interior
    # point shares the same x coordinate, so scipy's Qhull backend can't
    # build an initial simplex and raises QhullError. build_mesh must
    # translate that into a CompileError (compile.py's per-layer
    # warn-and-skip loop only catches CompileError), not let the raw
    # QhullError crash the compile.
    rgba = np.zeros((50, 1, 4), dtype=np.uint8)
    rgba[:, 0, 3] = 255
    with pytest.raises(CompileError):
        build_mesh(rgba, grid_step=1)


def test_density_scales_with_grid_step():
    rgba = ellipse_rgba(300, 300)
    dense = build_mesh(rgba, grid_step=16)
    coarse = build_mesh(rgba, grid_step=64)
    assert dense.vertices.shape[0] > coarse.vertices.shape[0]


def test_slot_density_table():
    assert grid_step_for_slot("hair_front") < grid_step_for_slot("misc")
    assert grid_step_for_slot("iris_l") > grid_step_for_slot("torso")


def test_disjoint_blobs_both_meshed():
    def draw(d):
        d.ellipse((10, 10, 70, 70), fill=(255, 0, 0, 255))
        # Small blob: fits between grid points at grid_step=20, so only its
        # own contour can contribute vertices — catches largest-contour-only bugs.
        d.ellipse((148, 148, 172, 172), fill=(255, 0, 0, 255))
    rgba = shape_rgba((200, 200), draw)
    mask = rgba[..., 3] > 8
    mesh = build_mesh(rgba, grid_step=20)
    centroids = mesh.vertices[mesh.triangles].mean(axis=1)
    in_first = (centroids[:, 0] < 100) & (centroids[:, 1] < 100)
    in_second = (centroids[:, 0] >= 100) & (centroids[:, 1] >= 100)
    assert in_first.any(), "no triangles cover the first blob"
    assert in_second.any(), "no triangles cover the second blob"
    assert tri_areas(mesh.vertices, mesh.triangles).sum() >= 0.75 * mask.sum()
