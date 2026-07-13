from dataclasses import dataclass

import cv2
import numpy as np
from scipy.spatial import Delaunay

from raig.compiler.errors import CompileError

_ALPHA_THRESHOLD = 8
_MIN_TRIANGLE_AREA = 0.5  # px^2

_SLOT_GRID_STEP: dict[str, int] = {
    "hair_front": 24, "hair_back": 24, "torso": 24, "face": 28,
    "iris_l": 64, "iris_r": 64, "eye_white_l": 64, "eye_white_r": 64,
}
_DEFAULT_GRID_STEP = 32


def grid_step_for_slot(slot: str) -> int:
    return _SLOT_GRID_STEP.get(slot, _DEFAULT_GRID_STEP)


@dataclass
class MeshResult:
    vertices: np.ndarray  # (N, 2) float32, local texture px
    uvs: np.ndarray  # (N, 2) float32 in [0, 1]
    triangles: np.ndarray  # (M, 3) int32


def build_mesh(rgba: np.ndarray, grid_step: int = _DEFAULT_GRID_STEP) -> MeshResult:
    mask = (rgba[..., 3] > _ALPHA_THRESHOLD).astype(np.uint8)
    if mask.sum() == 0:
        raise CompileError("layer has no opaque pixels")
    h, w = mask.shape

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea)
    perimeter = cv2.arcLength(contour, True)
    boundary = cv2.approxPolyDP(contour, 0.01 * perimeter, True).reshape(-1, 2)

    gx = np.arange(grid_step / 2.0, w, grid_step)
    gy = np.arange(grid_step / 2.0, h, grid_step)
    xx, yy = np.meshgrid(gx, gy)
    grid = np.stack([xx.ravel(), yy.ravel()], axis=1)
    inside = mask[grid[:, 1].astype(int), grid[:, 0].astype(int)] > 0
    points = np.vstack([boundary.astype(np.float64), grid[inside]])
    points = np.unique(np.round(points, 3), axis=0)
    if points.shape[0] < 3:
        raise CompileError("too few points to triangulate layer")

    tri = Delaunay(points)
    a = points[tri.simplices[:, 0]]
    b = points[tri.simplices[:, 1]]
    c = points[tri.simplices[:, 2]]
    areas = 0.5 * np.abs(np.cross(b - a, c - a))
    centroids = (a + b + c) / 3.0
    ix = np.clip(np.round(centroids[:, 0]).astype(int), 0, w - 1)
    iy = np.clip(np.round(centroids[:, 1]).astype(int), 0, h - 1)
    keep = (mask[iy, ix] > 0) & (areas > _MIN_TRIANGLE_AREA)
    triangles = tri.simplices[keep].astype(np.int32)
    if triangles.shape[0] == 0:
        raise CompileError("triangulation produced no interior triangles")

    vertices = points.astype(np.float32)
    uvs = (vertices / np.array([w, h], dtype=np.float32)).astype(np.float32)
    return MeshResult(vertices=vertices, uvs=uvs, triangles=triangles)
