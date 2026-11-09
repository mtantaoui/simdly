#!/usr/bin/env python3
"""Speed-up tables for the book, from criterion's results.

    cargo bench --features rayon
    python3 tools/bench_table.py [target/criterion]

Prints one Markdown table per category: simdly's speed-up over the plain
scalar loop (criterion's mean time, scalar / simdly) for every operation,
element type and length benchmarked. Then, from the same groups' `par` and
`rayon` results (`--features rayon`), two more tables per category: the
`par_` functions' speed-up over simdly on one thread, and over the rayon
loop with `std`. Missing results print as "–"; rows with no result at all
(`f64` for the operations benchmarked in `f32` only) are left out.
"""

import json
import sys
from pathlib import Path

CATEGORIES = {
    "Arithmetic": ["add", "sub", "mul", "div", "add_assign", "sub_assign", "mul_assign", "div_assign", "sub_scalar", "sub_scalar_assign", "div_scalar"],
    "Comparisons": ["less", "less_scalar", "equal", "equal_scalar"],
    "Math": ["abs", "abs_in_place", "sqrt", "sqrt_in_place", "acos", "asin", "atan", "atan_in_place", "atan2", "cbrt", "cbrt_in_place", "ceil", "ceil_in_place", "cos", "cos_in_place", "exp", "floor", "floor_in_place", "hypot", "hypot3", "hypot4", "ln", "sin", "sin_in_place", "tan", "tan_in_place"],
}
TYPES = ["f32", "f64"]
LENS = [64, 4_096, 262_144, 4_194_304]


def mean_ns(root: Path, group: str, impl: str, ty: str, n: int) -> float | None:
    path = root / group / f"{impl}_{ty}" / str(n) / "new" / "estimates.json"
    try:
        return json.loads(path.read_text())["mean"]["point_estimate"]
    except (OSError, KeyError, ValueError):
        return None


def cell(root: Path, group: str, ty: str, n: int, fast: str = "simdly", slow: str = "scalar") -> str:
    a = mean_ns(root, group, fast, ty, n)
    b = mean_ns(root, group, slow, ty, n)
    return f"{b / a:.2f}×" if a and b else "–"


def length(n: int) -> str:
    return f"{n:,}".replace(",", " ")


def table(title: str, columns: list[str], groups: list[str], cells) -> None:
    print(f"### {title}\n")
    print(f"| Operation | Type | {' | '.join(columns)} |")
    print("|---|---|" + "---:|" * len(columns))
    for group in groups:
        for ty in TYPES:
            row = cells(group, ty)
            if any(c != "–" for c in row):
                print(f"| `{group}` | `{ty}` | {' | '.join(row)} |")
    print()


def main() -> None:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "target/criterion")
    for title, groups in CATEGORIES.items():
        table(title, [length(n) for n in LENS], groups,
              lambda g, ty: [cell(root, g, ty, n) for n in LENS])
    for title, groups in CATEGORIES.items():
        for slow, over in (("simdly", "one thread"), ("rayon", "rayon with `std`")):
            table(f"{title}, parallel over {over}", [length(n) for n in LENS], groups,
                  lambda g, ty: [cell(root, g, ty, n, "par", slow) for n in LENS])


if __name__ == "__main__":
    main()
