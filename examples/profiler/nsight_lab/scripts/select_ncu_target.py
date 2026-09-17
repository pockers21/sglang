#!/usr/bin/env python3
import argparse
import csv
import json
import re
from pathlib import Path


def load_rows(path: Path):
    lines = path.read_text(encoding="utf-8").splitlines()
    try:
        start = next(index for index, line in enumerate(lines) if line.startswith("Time (%)"))
    except StopIteration as error:
        raise ValueError(f"Cannot find the Nsight Systems CSV header in {path}") from error
    rows = list(csv.DictReader(lines[start:]))
    if not rows:
        raise ValueError(f"No kernel rows found in {path}")
    return rows


def select(rows, limit=10):
    ranked = sorted(
        rows,
        key=lambda row: int(row.get("Total Time (ns)", 0)),
        reverse=True,
    )
    top = []
    for row in ranked[:limit]:
        top.append(
            {
                "name": row["Name"],
                "total_time_ms": int(row["Total Time (ns)"]) / 1e6,
                "instances": int(row["Instances"]),
                "time_percent": float(row["Time (%)"]),
            }
        )
    target = top[0]
    target["ncu_kernel_filter"] = "regex:^" + re.escape(target["name"]) + "$"
    return {"selected": target, "top_kernels": top}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--json", type=Path)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--print-filter", action="store_true")
    args = parser.parse_args()

    result = select(load_rows(args.input), args.limit)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if args.print_filter:
        print(result["selected"]["ncu_kernel_filter"])
    elif not args.json:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
