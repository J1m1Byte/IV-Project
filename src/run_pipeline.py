"""
run_pipeline.py  —  execute all project notebooks in order.

Usage
-----
    python run_pipeline.py            # run everything
    python run_pipeline.py --from 3   # start from step 3 (1-indexed)
    python run_pipeline.py --only model  # only data or model notebooks
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Pipeline order 
PIPELINE = [
    ("2.2 Feature — rand C",    "notebook/model/2.2-fc-rand-C-feature.ipynb"),
    ("2.0 Feature — rand A",    "notebook/model/2.0-fc-rand-A-feature.ipynb"),
    ("2.1 Feature — rand B",    "notebook/model/2.1-fc-rand-B-feature.ipynb"),
    ("2.3 Feature — rand D",    "notebook/model/2.3-fc-rand-D-feature.ipynb"),
]

# Helpers 
RESET  = "\033[0m"
GREEN  = "\033[32m"
RED    = "\033[31m"
YELLOW = "\033[33m"
BOLD   = "\033[1m"

def _fmt(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m}m {s:02d}s" if m else f"{s}s"

def run_notebook(path: Path) -> tuple[bool, float, str]:
    """Execute a notebook in-place via nbconvert. Returns (ok, elapsed, error)."""
    env = os.environ.copy()
    env["NOTEBOOK_STEM"] = path.stem 

    t0 = time.perf_counter()
    result = subprocess.run(
        [
            sys.executable, "-m", "nbconvert",
            "--to", "notebook",
            "--execute",
            "--inplace",
            "--ExecutePreprocessor.timeout=3600",
            "--ExecutePreprocessor.kernel_name=iv-project",
            str(path),
        ],
        capture_output=True,
        text=True,
        env=env,
    )
    elapsed = time.perf_counter() - t0
    if result.returncode != 0:
        error = (result.stderr or result.stdout).strip().splitlines()[-1]
        return False, elapsed, error
    return True, elapsed, ""

# Main
def main():
    parser = argparse.ArgumentParser(description="Run the IV project pipeline.")
    parser.add_argument("--from", dest="start", type=int, default=1,
                        help="Start from step N (1-indexed).")
    parser.add_argument("--only", choices=["data", "model"],
                        help="Run only 'data' or 'model' notebooks.")
    args = parser.parse_args()

    steps = [
        (i + 1, label, ROOT / nb)
        for i, (label, nb) in enumerate(PIPELINE)
    ]

    # Apply filters
    if args.only == "data":
        steps = [(n, l, p) for n, l, p in steps if "data" in str(p)]
    elif args.only == "model":
        steps = [(n, l, p) for n, l, p in steps if "model" in str(p)]

    steps = [(n, l, p) for n, l, p in steps if n >= args.start]

    if not steps:
        print("No notebooks matched the given filters.")
        sys.exit(0)

    print(f"\n{BOLD}IV Project Pipeline{RESET}  ({len(steps)} notebook(s))\n")
    print(f"  {'#':<4} {'Notebook':<35} {'Status':<10} {'Time'}")
    print(f"  {'-'*4} {'-'*35} {'-'*10} {'-'*8}")

    results = []
    total_start = time.perf_counter()

    for step, label, path in steps:
        if not path.exists():
            print(f"  {step:<4} {label:<35} {YELLOW}MISSING{RESET}")
            results.append((step, label, "missing", 0.0, "file not found"))
            continue

        print(f"  {step:<4} {label:<35} running…", end="", flush=True)
        ok, elapsed, error = run_notebook(path)

        status = f"{GREEN}OK{RESET}" if ok else f"{RED}FAILED{RESET}"
        print(f"\r  {step:<4} {label:<35} {status:<18} {_fmt(elapsed)}")

        if not ok:
            print(f"       {RED}↳ {error}{RESET}")

        results.append((step, label, "ok" if ok else "failed", elapsed, error))

    # Summary
    total = time.perf_counter() - total_start
    n_ok     = sum(1 for *_, s, _, __ in results if s == "ok")
    n_failed = sum(1 for *_, s, _, __ in results if s == "failed")
    n_skip   = sum(1 for *_, s, _, __ in results if s == "missing")

    print(f"\n  {BOLD}{'─'*58}{RESET}")
    print(f"  {BOLD}Total: {_fmt(total)}{RESET}   "
          f"{GREEN}{n_ok} passed{RESET}  "
          f"{RED}{n_failed} failed{RESET}  "
          f"{YELLOW}{n_skip} missing{RESET}\n")

    sys.exit(1 if n_failed else 0)


if __name__ == "__main__":
    main()
