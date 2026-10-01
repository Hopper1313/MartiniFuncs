"""python -m MartiniFuncs ...: the postprocessing CLI (MartiniFuncs.workflows.cli)."""
import sys

from MartiniFuncs.workflows.cli import main

if __name__ == "__main__":
    sys.argv[0] = "python -m MartiniFuncs"
    raise SystemExit(main())
