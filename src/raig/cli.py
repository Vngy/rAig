import argparse


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="raig", description="Auto-rigging for 2D vtuber avatars"
    )
    sub = p.add_subparsers(dest="command", required=True)

    c = sub.add_parser("compile", help="Compile a layered PSD into a .raig rig")
    c.add_argument("psd", help="Path to layered PSD")
    c.add_argument("-o", "--output", default=None, help="Output .raig path")
    c.add_argument("--overrides", default=None, help="Path to overrides.toml")

    r = sub.add_parser("run", help="Live preview a .raig rig from webcam")
    r.add_argument("rig", help="Path to .raig file")
    r.add_argument("--replay", default=None, help="Landmark JSONL instead of webcam")
    r.add_argument("--camera", type=int, default=0, help="Webcam index")

    d = sub.add_parser("debug", help="Write debug sheet PNGs for a .raig rig")
    d.add_argument("rig", help="Path to .raig file")
    d.add_argument("-o", "--outdir", default="debug_sheet", help="Output directory")

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "compile":
        from raig.compiler.compile import compile_command

        return compile_command(args)
    if args.command == "run":
        from raig.render.window import run_command

        return run_command(args)
    if args.command == "debug":
        from raig.debugsheet import debug_command

        return debug_command(args)
    return 2
