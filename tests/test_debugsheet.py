from dataclasses import replace

import numpy as np
from PIL import Image

from raig.cli import main
from raig.core.rig import Rig, save_rig
from raig.debugsheet import write_debug_sheet

BONE_LINE_RGB = (255, 200, 0)  # cv2 draws BGR (0, 200, 255); saved PNG is RGB
WIREFRAME_RGB = (0, 255, 0)


def _has_color(png_path, rgb) -> bool:
    img = np.asarray(Image.open(png_path).convert("RGB"))
    return bool(np.any(np.all(img == rgb, axis=-1)))


def test_writes_overview_and_per_layer_pngs(mini_rig, tmp_path):
    written = write_debug_sheet(mini_rig, tmp_path)
    assert len(written) == 1 + len(mini_rig.layers)  # overview + 13 layers
    for p in written:
        assert p.exists() and p.stat().st_size > 0
    assert (tmp_path / "00_overview.png").exists()


def test_overlay_colors_survive_to_disk(mini_rig, tmp_path):
    written = write_debug_sheet(mini_rig, tmp_path)
    # skeleton overlay on the overview: exact orange bone-line pixels
    assert _has_color(written[0], BONE_LINE_RGB)
    # mesh wireframe on per-layer sheets: exact pure-green pixels
    assert any(_has_color(p, WIREFRAME_RGB) for p in written[1:])


def test_japanese_layer_name_gets_ascii_filename(mini_rig, tmp_path):
    layer = replace(mini_rig.layers[0], layer_name="前髪")
    rig = Rig(
        canvas_size=mini_rig.canvas_size,
        layers=[layer],
        bones=mini_rig.bones,
        deformers=[],
    )
    written = write_debug_sheet(rig, tmp_path)
    layer_png = written[1]
    assert layer_png.name.isascii()
    assert layer_png.exists()


def test_cli_debug_command(mini_rig, tmp_path):
    rig_path = tmp_path / "mini.raig"
    save_rig(mini_rig, rig_path)
    outdir = tmp_path / "sheet"
    assert main(["debug", str(rig_path), "-o", str(outdir)]) == 0
    assert len(list(outdir.glob("*.png"))) == 1 + len(mini_rig.layers)
