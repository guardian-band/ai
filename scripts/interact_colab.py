"""Interactively run the GraphSAGE smoke command in a Colab console."""

import argparse
import shutil

import pexpect


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
    child = pexpect.spawn(
        executable,
        ["console", "-s", args.session],
        encoding="utf-8",
    )
    try:
        child.expect([r"\$", r"#", pexpect.TIMEOUT], timeout=10)
        print("Connected to Colab console.")
        child.sendline("cd /content && tar -xvf payload.tar")
        child.expect([r"\$", r"#", pexpect.TIMEOUT], timeout=10)
        print(child.before)
        child.sendline("python3 scripts/colab_graphsage_smoke_training.py")
        child.expect([r"\$", r"#", pexpect.TIMEOUT], timeout=60)
        print(child.before)
        child.sendline("exit")
    finally:
        child.close()


if __name__ == "__main__":
    main()
