"""Explicit entry point for commands outside P1."""

import sys


def main() -> int:
    command = sys.argv[1] if len(sys.argv) > 1 else "command"
    print(f"{command}: not implemented in P1; see prompt.md for P2-P7 scope.")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
