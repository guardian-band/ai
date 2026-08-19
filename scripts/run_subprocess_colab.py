"""Run the GraphSAGE smoke command in a Colab console subprocess."""

import argparse
import shutil
import subprocess


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", required=True, help="Colab console session identifier")
    parser.add_argument(
        "--colab-executable",
        help="Path to the colab executable (defaults to shutil.which('colab'))",
    )
    return parser.parse_args()


def _resolve_executable(explicit: str | None) -> str:
    executable = explicit or shutil.which("colab")
    if not executable:
        raise SystemExit("colab executable not found; pass --colab-executable or add colab to PATH")
    return executable


def main() -> None:
    args = _parse_args()
    executable = _resolve_executable(args.colab_executable)
    process = subprocess.Popen(
        [executable, "console", "-s", args.session],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    output, _ = process.communicate(
        input="cd /content\ntar -xvf payload.tar\n"
        "python3 scripts/colab_graphsage_smoke_training.py\nexit\n"
    )
    print(output)


if __name__ == "__main__":
    main()
