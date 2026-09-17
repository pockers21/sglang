#!/usr/bin/env python3
import argparse
import json
import statistics
from pathlib import Path


METRICS = (
    "prefill_latency",
    "prefill_throughput",
    "median_decode_latency",
    "median_decode_throughput",
    "total_latency",
    "overall_throughput",
)


def load_jsonl(path: Path):
    rows = []
    with path.open(encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"No benchmark records found in {path}")
    return rows


def describe(rows):
    result = {"runs": len(rows)}
    for metric in METRICS:
        values = [float(row[metric]) for row in rows]
        result[metric] = {
            "mean": statistics.fmean(values),
            "median": statistics.median(values),
            "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
            "min": min(values),
            "max": max(values),
        }
    return result


def milliseconds(value):
    return value * 1000.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--disabled", type=Path, required=True)
    parser.add_argument("--full", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args()

    summary = {
        "disabled": describe(load_jsonl(args.disabled)),
        "full": describe(load_jsonl(args.full)),
    }
    eager = summary["disabled"]["median_decode_latency"]["median"]
    graph = summary["full"]["median_decode_latency"]["median"]
    summary["decode_speedup"] = eager / graph
    summary["decode_latency_reduction_percent"] = (1.0 - graph / eager) * 100.0

    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# CUDA Graph A/B Summary",
        "",
        "| Mode | Runs | Decode median | Decode mean | Decode stdev | Decode throughput | Total latency |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for mode in ("disabled", "full"):
        data = summary[mode]
        decode = data["median_decode_latency"]
        lines.append(
            f"| `{mode}` | {data['runs']} | {milliseconds(decode['median']):.3f} ms | "
            f"{milliseconds(decode['mean']):.3f} ms | {milliseconds(decode['stdev']):.3f} ms | "
            f"{data['median_decode_throughput']['median']:.2f} tok/s | "
            f"{milliseconds(data['total_latency']['median']):.3f} ms |"
        )
    lines.extend(
        [
            "",
            f"Decode latency speedup: **{summary['decode_speedup']:.3f}x**.",
            "",
            "Decode latency reduction: "
            f"**{summary['decode_latency_reduction_percent']:.2f}%**.",
            "",
            "Timing comes from unprofiled `sglang.benchmark.one_batch` runs. "
            "Use the paired Nsight Systems traces to explain the result, not to report latency.",
        ]
    )
    args.markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
