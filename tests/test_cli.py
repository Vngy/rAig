import pytest

from raig.cli import build_parser


def test_help_lists_subcommands(capsys):
    parser = build_parser()
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    for cmd in ("compile", "run", "debug"):
        assert cmd in out


def test_compile_args():
    args = build_parser().parse_args(
        ["compile", "model.psd", "-o", "out.raig", "--overrides", "ov.toml"]
    )
    assert args.command == "compile"
    assert args.psd == "model.psd"
    assert args.output == "out.raig"
    assert args.overrides == "ov.toml"


def test_run_args_defaults():
    args = build_parser().parse_args(["run", "model.raig"])
    assert args.command == "run"
    assert args.rig == "model.raig"
    assert args.replay is None
    assert args.camera == 0


def test_debug_args():
    args = build_parser().parse_args(["debug", "model.raig"])
    assert args.command == "debug"
    assert args.outdir == "debug_sheet"
