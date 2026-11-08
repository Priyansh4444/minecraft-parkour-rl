"""Run local formatting, lint, typing and test checks."""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def run(command: list[str]) -> None:
    print(f"\n$ {' '.join(command)}", flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fix", action="store_true", help="apply formatting and safe lint fixes first"
    )
    args = parser.parse_args()
    try:
        if args.fix:
            run(["ruff", "check", ".", "--fix"])
            run(["ruff", "format", "."])
            run(["npm", "run", "format", "--prefix", "bridge"])
        run(["ruff", "format", "--check", "."])
        run(["ruff", "check", "."])
        run(["ty", "check", "--error-on-warning"])
        run(["npm", "run", "check", "--prefix", "bridge"])
        run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"])
        run(["npm", "test", "--prefix", "bridge"])
    except subprocess.CalledProcessError as error:
        raise SystemExit(error.returncode) from error
    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
