"""
Run jcvi.compara.synteny scan for all 6 subgenome pairs.
Input:  <pair>.filtered  +  <q>.bed  +  <s>.bed
Output: <pair>.anchors
"""

import os
import subprocess
import sys
import time

PY = r"G:\iub_research\Muhammad Ali and Faisal\vscoder\.venv\Scripts\python.exe"
DIR = r"G:\iub_research\kinesin7_synteny\2026-04-14"

PAIRS = [
    ("Ga", "GhA"),
    ("Ga", "GbA"),
    ("GhA", "GbA"),
    ("Gr", "GhD"),
    ("Gr", "GbD"),
    ("GhD", "GbD"),
]


def count_lines(file_path):
    with open(file_path, encoding="utf-8") as handle:
        return sum(1 for _ in handle)


def main():
    os.chdir(DIR)

    for q, s in PAIRS:
        filtered = os.path.join(DIR, f"{q}_{s}.filtered")
        anchors = os.path.join(DIR, f"{q}_{s}.anchors")
        qbed = os.path.join(DIR, f"{q}.bed")
        sbed = os.path.join(DIR, f"{s}.bed")

        if os.path.exists(anchors):
            print(f"[SKIP] {q}_{s}.anchors already exists")
            continue

        if not os.path.exists(filtered):
            print(f"[MISSING] {filtered}")
            sys.exit(1)

        print(f"\n{'=' * 55}")
        print(f"  synteny scan: {q} vs {s}  ->  {q}_{s}.anchors")
        print(f"{'=' * 55}")
        t0 = time.time()

        cmd = [
            PY,
            "-m",
            "jcvi.compara.synteny",
            "scan",
            filtered,
            anchors,
            "--dist",
            "20",
            "--min_size",
            "4",
            "--qbed",
            qbed,
            "--sbed",
            sbed,
        ]
        result = subprocess.run(cmd, cwd=DIR, check=False)

        if result.returncode != 0:
            print(f"  ERROR: synteny scan failed for {q} vs {s} (exit {result.returncode})")
            sys.exit(1)

        elapsed = time.time() - t0
        if os.path.exists(anchors):
            size = os.path.getsize(anchors)
            lines = count_lines(anchors)
            print(f"  Done in {elapsed:.0f}s  ->  {q}_{s}.anchors  ({size / 1e3:.1f} KB, {lines} lines)")
        else:
            print(f"  WARNING: {anchors} not created - check output above")

    print("\n\nAll 6 synteny scans done.")
    print("Anchors files:")
    for q, s in PAIRS:
        anchors = os.path.join(DIR, f"{q}_{s}.anchors")
        if os.path.exists(anchors):
            lines = count_lines(anchors)
            print(f"  {q}_{s}.anchors  ({lines} lines)")


if __name__ == "__main__":
    main()