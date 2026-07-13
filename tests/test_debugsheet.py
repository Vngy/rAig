from raig.cli import main
from raig.core.rig import save_rig
from raig.debugsheet import write_debug_sheet


def test_writes_overview_and_per_layer_pngs(mini_rig, tmp_path):
    written = write_debug_sheet(mini_rig, tmp_path)
    assert len(written) == 1 + len(mini_rig.layers)  # overview + 13 layers
    for p in written:
        assert p.exists() and p.stat().st_size > 0
    assert (tmp_path / "00_overview.png").exists()


def test_cli_debug_command(mini_rig, tmp_path):
    rig_path = tmp_path / "mini.raig"
    save_rig(mini_rig, rig_path)
    outdir = tmp_path / "sheet"
    assert main(["debug", str(rig_path), "-o", str(outdir)]) == 0
    assert len(list(outdir.glob("*.png"))) == 1 + len(mini_rig.layers)
