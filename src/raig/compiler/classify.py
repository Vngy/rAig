import re
import tomllib
from dataclasses import dataclass
from pathlib import Path

from raig.compiler.psd_ingest import LayerRecord

SLOTS: frozenset[str] = frozenset({
    "hair_front", "hair_back", "face",
    "eye_white_l", "iris_l", "eye_white_r", "iris_r",
    "brow_l", "brow_r", "mouth", "head_misc",
    "torso", "arm_l", "arm_r", "leg_l", "leg_r", "misc",
})

HEAD_SLOTS: frozenset[str] = SLOTS - {"torso", "arm_l", "arm_r", "leg_l", "leg_r", "misc"}

# (base, keywords) in priority order: first substring hit wins.
# Bases in SIDED get _l/_r appended after side resolution.
_BASE_RULES: list[tuple[str, list[str]]] = [
    ("hair_front", ["hair_front", "fronthair", "bangs", "fringe", "前髪"]),
    ("hair_back", ["hair_back", "backhair", "後ろ髪", "後髪"]),
    ("iris", ["iris", "pupil", "瞳", "目玉"]),
    ("eye_white", ["eye_white", "eyewhite", "sclera", "白目"]),
    ("brow", ["brow", "眉"]),
    ("mouth", ["mouth", "lips", "口"]),
    ("face", ["face", "顔"]),
    ("hand", ["hand", "手"]),
    ("arm", ["arm", "腕"]),
    ("leg", ["leg", "foot", "脚", "足"]),
    ("torso", ["torso", "body", "chest", "胴"]),
    ("head_misc", ["head", "頭"]),
]

_SIDED = {"iris", "eye_white", "brow", "arm", "leg", "hand"}
_LEFT_TOKENS = {"l", "left"}
_RIGHT_TOKENS = {"r", "right"}


@dataclass
class Classified:
    record: LayerRecord
    slot: str
    source: str  # "override" | "name" | "inherit" | "geometry"


def _normalize(name: str) -> str:
    return re.sub(r"[\s\-]+", "", name.lower())


def _match_base(name: str) -> str | None:
    norm = _normalize(name)
    for base, keywords in _BASE_RULES:
        if any(k in norm for k in keywords):
            return base
    return None


def _detect_side(name: str) -> str | None:
    if "左" in name:
        return "l"
    if "右" in name:
        return "r"
    tokens = set(re.split(r"[_\s\-]+", name.lower()))
    if tokens & _LEFT_TOKENS:
        return "l"
    if tokens & _RIGHT_TOKENS:
        return "r"
    return None


def _bbox_center(record: LayerRecord) -> tuple[float, float]:
    h, w = record.rgba.shape[:2]
    return record.offset[0] + w / 2.0, record.offset[1] + h / 2.0


def _resolve(base: str, record: LayerRecord, canvas_size: tuple[int, int]) -> str:
    if base == "hand":
        base = "arm"
    if base not in _SIDED:
        return base
    side = _detect_side(record.name)
    if side is None:
        cx, _ = _bbox_center(record)
        side = "l" if cx < canvas_size[0] / 2.0 else "r"
    return f"{base}_{side}"


def _geometry_slot(record: LayerRecord, canvas_size: tuple[int, int]) -> str:
    _, cy = _bbox_center(record)
    return "head_misc" if cy < 0.4 * canvas_size[1] else "misc"


def classify_layers(
    records: list[LayerRecord],
    canvas_size: tuple[int, int],
    overrides: dict[str, str] | None = None,
) -> list[Classified]:
    overrides = overrides or {}
    out: list[Classified] = []
    for record in records:
        if record.name in overrides:
            out.append(Classified(record, overrides[record.name], "override"))
            continue
        base = _match_base(record.name)
        if base is not None:
            out.append(Classified(record, _resolve(base, record, canvas_size), "name"))
            continue
        inherited = None
        for group_name in reversed(record.group_path):  # nearest ancestor first
            gbase = _match_base(group_name)
            if gbase is not None:
                # side may live on the group name ("arm_l"), not the layer
                side = _detect_side(group_name)
                fake = LayerRecord(
                    name=group_name, group_path=(), z_index=record.z_index,
                    offset=record.offset, rgba=record.rgba,
                ) if side is not None else record
                inherited = _resolve(gbase, fake, canvas_size)
                break
        if inherited is not None:
            out.append(Classified(record, inherited, "inherit"))
            continue
        out.append(Classified(record, _geometry_slot(record, canvas_size), "geometry"))
    return out


def load_overrides(path: str | Path) -> dict[str, str]:
    with open(path, "rb") as f:
        data = tomllib.load(f)
    slots = data.get("slots", {})
    for layer_name, slot in slots.items():
        if slot not in SLOTS:
            raise ValueError(
                f"overrides: layer {layer_name!r} maps to unknown slot {slot!r}; "
                f"valid slots: {sorted(SLOTS)}"
            )
    return dict(slots)
