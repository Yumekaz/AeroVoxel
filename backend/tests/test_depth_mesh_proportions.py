"""Image-space proportions must survive the neural depth-to-mesh stage."""

import numpy as np
import pytest

from app.recon.depth_mesh import depth_map_to_mesh, validate_mesh


@pytest.mark.parametrize("height,width", [(80, 160), (160, 80), (96, 96)])
def test_rectangular_foreground_keeps_pixel_proportions(height, width):
    mask = np.zeros((height, width), dtype=bool)
    mask[10:height-10, 15:width-15] = True
    depth = np.linspace(0, 1, height * width).reshape(height, width)
    mesh = depth_map_to_mesh(depth, mask, max_size=192)
    expected_ratio = (width - 31) / (height - 21)
    assert mesh.extents[0] / mesh.extents[1] == pytest.approx(expected_ratio)
    assert validate_mesh(mesh)["valid"]
    assert mesh.is_watertight


def test_minimum_mesh_raster_height_does_not_stretch_a_thin_image():
    height, width = 32, 400
    mask = np.ones((height, width), dtype=bool)
    depth = np.linspace(0, 1, height * width).reshape(height, width)
    mesh = depth_map_to_mesh(depth, mask, max_size=96)
    assert mesh.extents[0] / mesh.extents[1] == pytest.approx((width - 1) / (height - 1))
