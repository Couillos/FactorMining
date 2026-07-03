#!/usr/bin/env python3
"""Legacy entry point for the FactorMining production pipeline.

Historically this file contained the full pipeline (evolution →
backtest → validation → reporting). That logic now lives in
:mod:`factor_mining.cli` and is exposed as the ``factor-mining``
console script via ``[project.scripts]`` in ``pyproject.toml``::

    factor-mining --config config/default.yaml --seed 42 --output-dir ./output

This file is preserved as a thin shim so existing invocations of
``python run_pipeline.py ...`` continue to work unchanged — every CLI
flag is forwarded verbatim to :func:`factor_mining.cli.main`.

The pipeline source has a single home (``src/factor_mining/cli.py``);
this file deliberately contains no duplicated logic.
"""

from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> int:
    """Delegate to :func:`factor_mining.cli.main`.

    ``argv`` defaults to ``sys.argv[1:]`` (mirroring argparse's default
    behaviour) so the caller can drop in either a literal argument list
    or rely on the process command line.
    """
    from factor_mining.cli import main as cli_main

    return cli_main(argv if argv is not None else sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
