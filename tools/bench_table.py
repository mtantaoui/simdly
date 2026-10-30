#!/usr/bin/env python3
"""Speed-up tables for the book, from criterion's results.

    cargo bench
    python3 tools/bench_table.py [target/criterion]

Prints one Markdown table per category: simdly's speed-up over the plain
scalar loop (criterion's mean time, scalar / simdly) for every operation,
element type and length benchmarked. Missing results print as "–".
"""

import json
import sys
from pathlib import Path

CATEGORIES = {
    "Arithmetic": ["add", "sub", "mul", "div", "add_assign", "sub_assign", "mul_assign", "div_assign"],
    "Comparisons": ["less", "less_scalar", "equal", "equal_scalar"],
    "Math": ["abs", "abs_in_place", "sqrt", "sqrt_in_place", "acos", "asin", "atan", "atan_in_place", "atan2", "cbrt", "cbrt_in_place", "exp", "floor", "floor_in_place", "hypot", "ln", "sin", "sin_in_place"],
}
TYPES = ["f32", "f64"]
LENS = [64, 4_096, 262_144, 4_194_304]


def mean_ns(root: Path, group: str, impl: str, ty: str, n: int) -> float | None:
    path = root / group / f"{impl}_{ty}" / str(n) / "new" / "estimates.json"
    try:
        return json.loads(path.read_text())["mean"]["point_estimate"]
    except (OSError, KeyError, ValueError):
        return None


def cell(root: Path, group: str, ty: str, n: int) -> str:
    simdly = mean_ns(root, group, "simdly", ty, n)
    scalar = mean_ns(root, group, "scalar", ty, n)
    return f"{scalar / simdly:.2f}×" if simdly and scalar else "–"


def main() -> None:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "target/criterion")
    header = " | ".join(f"{n:,}".replace(",", " ") for n in LENS)
    for title, groups in CATEGORIES.items():
        print(f"### {title}\n")
        print(f"| Operation | Type | {header} |")
        print("|---|---|" + "---:|" * len(LENS))
        for group in groups:
            for ty in TYPES:
                cells = " | ".join(cell(root, group, ty, n) for n in LENS)
                print(f"| `{group}` | `{ty}` | {cells} |")
        print()


if __name__ == "__main__":
    main()
