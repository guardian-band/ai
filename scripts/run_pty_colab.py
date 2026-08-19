"""Run the GraphSAGE smoke command in an interactive Colab console."""

import argparse
import os
import pty
import select
import shutil
import time


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
    pid, fd = pty.fork()
    if pid == 0:
        os.execv(executable, [executable, "console", "-s", args.session])

    try:
        time.sleep(3)
        os.write(fd, b"cd /content && tar -xvf payload.tar\n")
        time.sleep(2)
        os.write(fd, b"python3 scripts/colab_graphsage_smoke_training.py\n")

        start_time = time.time()
        while time.time() - start_time < 60:
            readable, _, _ = select.select([fd], [], [], 1.0)
            if fd not in readable:
                continue
            try:
                data = os.read(fd, 1024)
            except OSError:
                break
            if not data:
                break
            print(data.decode("utf-8", errors="replace"), end="", flush=True)
            if b"Smoke training successfully completed." in data:
                break
    finally:
        try:
            os.write(fd, b"exit\n")
        except OSError:
            pass


if __name__ == "__main__":
    main()
