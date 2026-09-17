#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


REQUIRED = (
    "batch_size",
    "input_len",
    "output_len",
    "prefill_latency",
    "prefill_throughput",
    "median_decode_latency",
    "median_decode_throughput",
    "total_latency",
    "overall_throughput",
)


def load_rows(path: Path):
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        missing = [field for field in REQUIRED if field not in row]
        if missing:
            raise ValueError(f"{path}:{number} is missing {', '.join(missing)}")
        rows.append(row)
    if not rows:
        raise ValueError(f"No benchmark records found in {path}")
    return sorted(rows, key=lambda row: (row["batch_size"], row["input_len"], row["output_len"]))


def summarize(rows):
    normalized = []
    for row in rows:
        normalized.append(
            {
                "batch_size": int(row["batch_size"]),
                "input_len": int(row["input_len"]),
                "output_len": int(row["output_len"]),
                "prefill_latency_ms": float(row["prefill_latency"]) * 1000,
                "prefill_throughput": float(row["prefill_throughput"]),
                "decode_latency_ms": float(row["median_decode_latency"]) * 1000,
                "decode_throughput": float(row["median_decode_throughput"]),
                "total_latency_ms": float(row["total_latency"]) * 1000,
                "overall_throughput": float(row["overall_throughput"]),
            }
        )
    return {
        "points": normalized,
        "best_prefill_throughput": max(normalized, key=lambda row: row["prefill_throughput"]),
        "best_decode_throughput": max(normalized, key=lambda row: row["decode_throughput"]),
        "best_overall_throughput": max(normalized, key=lambda row: row["overall_throughput"]),
    }


def render_markdown(summary):
    lines = [
        "# Offline Shape Sweep",
        "",
        "| Batch | Input | Output | Prefill | Prefill throughput | Decode/token | Decode throughput | Overall throughput |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["points"]:
        lines.append(
            f"| {row['batch_size']} | {row['input_len']} | {row['output_len']} | "
            f"{row['prefill_latency_ms']:.2f} ms | {row['prefill_throughput']:.1f} tok/s | "
            f"{row['decode_latency_ms']:.3f} ms | {row['decode_throughput']:.1f} tok/s | "
            f"{row['overall_throughput']:.1f} tok/s |"
        )
    best = summary["best_overall_throughput"]
    lines.extend(
        [
            "",
            "Best measured overall-throughput point: "
            f"**batch={best['batch_size']}, input={best['input_len']}, output={best['output_len']}**, "
            f"at **{best['overall_throughput']:.1f} tok/s**.",
            "",
            "This sweep describes the low-level offline runner. It does not include HTTP, scheduling, "
            "queueing, or multi-request serving overhead.",
        ]
    )
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    args = parser.parse_args()

    summary = summarize(load_rows(args.input))
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    args.markdown.write_text(render_markdown(summary), encoding="utf-8")


if __name__ == "__main__":
    main()
