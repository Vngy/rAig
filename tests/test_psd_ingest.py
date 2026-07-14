import numpy as np
from PIL import Image as PILImage
from PIL import ImageDraw
from pytoshop import enums
from pytoshop.user import nested_layers

from raig.compiler import psd_ingest
from raig.compiler.psd_ingest import load_layers


def test_loads_all_pixel_layers(mini_psd_path):
    records, canvas = load_layers(mini_psd_path)
    assert canvas == (1024, 1024)
    names = {r.name for r in records}
    assert names == {
        "hair_front", "iris_l", "eye_white_l", "iris_r", "eye_white_r",
        "mouth", "face", "hair_back",
        "arm_l", "arm_r", "torso", "leg_l", "leg_r",
    }


def test_group_paths(mini_psd_path):
    records, _ = load_layers(mini_psd_path)
    by_name = {r.name: r for r in records}
    assert by_name["face"].group_path == ("head",)
    assert by_name["torso"].group_path == ("body",)


def test_rgba_and_offset(mini_psd_path):
    records, _ = load_layers(mini_psd_path)
    by_name = {r.name: r for r in records}
    face = by_name["face"]
    assert face.rgba.dtype == np.uint8
    assert face.rgba.shape[2] == 4
    assert face.rgba[..., 3].max() == 255
    left, top = face.offset
    assert 380 <= left <= 400 and 120 <= top <= 140  # near (392, 130)


def test_z_order_is_draw_order(mini_psd_path):
    records, _ = load_layers(mini_psd_path)
    z = {r.name: r.z_index for r in records}
    # painter order semantics: bigger z draws on top
    assert z["iris_l"] > z["eye_white_l"] > z["face"] > z["hair_back"]
    assert z["hair_front"] > z["face"]
    assert z["torso"] > z["leg_l"]
    assert z["face"] > z["torso"]  # head group is above body group
    assert sorted(z.values()) == list(range(13))


def _rect_layer(name: str, box: tuple[int, int, int, int], color, size=64):
    # Mirrors tests/fixtures/make_test_psd.py's shape_layer helper.
    img = PILImage.new("RGBA", (size, size), (0, 0, 0, 0))
    ImageDraw.Draw(img).rectangle(box, fill=color)
    arr = np.array(img)
    ys, xs = np.nonzero(arr[..., 3])
    top, left = int(ys.min()), int(xs.min())
    bottom, right = int(ys.max()) + 1, int(xs.max()) + 1
    crop = np.ascontiguousarray(arr[top:bottom, left:right])
    channels = {0: crop[..., 0], 1: crop[..., 1], 2: crop[..., 2], -1: crop[..., 3]}
    return nested_layers.Image(
        name=name, top=top, left=left, bottom=bottom, right=right, channels=channels
    )


def _write_psd(layers, path, size=64):
    psd = nested_layers.nested_layers_to_psd(
        layers,
        color_mode=enums.ColorMode.rgb,
        size=(size, size),
        # see tests/fixtures/make_test_psd.py: zip avoids the uncompiled
        # packbits (RLE) extension on this Python version.
        compression=enums.Compression.zip,
    )
    with open(path, "wb") as f:
        psd.write(f)


def test_duplicate_layer_names_are_uniquified(tmp_path):
    layers = [
        _rect_layer("hair", (4, 4, 20, 20), (120, 80, 160, 255)),
        _rect_layer("hair", (30, 30, 46, 46), (255, 0, 0, 255)),
    ]
    path = tmp_path / "dup.psd"
    _write_psd(layers, path)

    records, _ = load_layers(path)
    names = [r.name for r in records]
    assert names == ["hair", "hair#2"]
    assert len({r.name for r in records}) == 2  # both layers survive distinctly
    # both keep their own distinct pixel content (not one silently dropped).
    # psd-tools iterates file storage order (bottom-most/last-listed first),
    # so "hair" (first encountered, first uniquified name) is the pure-red
    # rect listed last above, and "hair#2" is the purple one listed first.
    by_name = {r.name: r for r in records}
    assert tuple(by_name["hair"].rgba[0, 0]) == (255, 0, 0, 255)
    assert tuple(by_name["hair#2"].rgba[0, 0]) == (120, 80, 160, 255)


class _FakeLayer:
    """Minimal stand-in for a psd-tools layer object, exposing only what
    load_layers touches (name, kind, parent, is_group(), composite(),
    left/top). Used because generating a real adjustment/type layer via
    pytoshop is impractical (pytoshop's nested_layers only builds group and
    raster/Image layers)."""

    def __init__(self, name, kind, rgba=None, parent=None, offset=(0, 0)):
        self.name = name
        self.kind = kind
        self.parent = parent
        self.left, self.top = offset
        self._rgba = rgba

    def is_group(self):
        return False

    def composite(self):
        return PILImage.fromarray(self._rgba, "RGBA")


class _FakePSD:
    def __init__(self, layers, size=(64, 64)):
        self._layers = layers
        self.width, self.height = size

    def descendants(self):
        return self._layers


def test_non_pixel_layers_skipped_with_stderr_warning(monkeypatch, capsys):
    pixel = _FakeLayer("face", "pixel", rgba=np.full((4, 4, 4), 255, np.uint8))
    adjustment = _FakeLayer("Levels 1", "levels")
    text = _FakeLayer("Title", "type")
    fake_psd = _FakePSD([pixel, adjustment, text])

    class _FakePSDImage:
        @staticmethod
        def open(_path):
            return fake_psd

    monkeypatch.setattr(psd_ingest, "PSDImage", _FakePSDImage)
    records, canvas = psd_ingest.load_layers("unused.psd")

    assert [r.name for r in records] == ["face"]
    assert canvas == (64, 64)
    err = capsys.readouterr().err
    assert "Levels 1" in err
    assert "Title" in err
