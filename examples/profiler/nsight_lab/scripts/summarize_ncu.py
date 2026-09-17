#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path


REQUIRED_METRICS = (
    "Duration",
    "Memory Throughput",
    "DRAM Throughput",
    "L1/TEX Cache Throughput",
    "L2 Cache Throughput",
    "Compute (SM) Throughput",
)
OPTIONAL_METRICS = (
    "Mem Busy",
    "L1/TEX Hit Rate",
    "L2 Hit Rate",
    "One or More Eligible",
    "Issued Warp Per Scheduler",
    "No Eligible",
    "Active Warps Per Scheduler",
    "Eligible Warps Per Scheduler",
    "Theoretical Occupancy",
    "Achieved Occupancy",
    "Warp Cycles Per Issued Instruction",
    "Branch Efficiency",
)


def parse_metrics(text):
    values = {}
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("Memory Throughput") and "Tbyte/s" in stripped:
            match = re.search(r"\s([0-9]+(?:\.[0-9]+)?)$", stripped)
            if match:
                values["Memory Throughput (Tbyte/s)"] = float(match.group(1))
            continue
        for metric in REQUIRED_METRICS + OPTIONAL_METRICS:
            if metric not in values and stripped.startswith(metric):
                match = re.search(r"\s([0-9]+(?:\.[0-9]+)?)$", stripped)
                if match:
                    values[metric] = float(match.group(1))
                break
    missing = [metric for metric in REQUIRED_METRICS if metric not in values]
    if missing:
        raise ValueError(f"NCU details are missing metrics: {', '.join(missing)}")
    scoreboard = re.search(
        r"spends\s+([0-9]+(?:\.[0-9]+)?) cycles being stalled waiting for a scoreboard dependency"
        r".*?represents about\s+([0-9]+(?:\.[0-9]+)?)%",
        text,
        re.DOTALL,
    )
    if scoreboard:
        values["Long Scoreboard Stall Cycles"] = float(scoreboard.group(1))
        values["Long Scoreboard Stall Share"] = float(scoreboard.group(2))
    return values


def parse_sections(text):
    sections = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("Section:"):
            section = stripped.split(":", 1)[1].strip()
            if section and section not in sections:
                sections.append(section)
    return sections


def classify(metrics):
    memory = metrics["Memory Throughput"]
    compute = metrics["Compute (SM) Throughput"]
    if memory >= 60 and memory >= compute * 1.5:
        return "memory-throughput-dominant"
    if compute >= 60 and compute >= memory * 1.25:
        return "compute-throughput-dominant"
    if compute >= 60 and memory >= 60:
        return "balanced-high-utilization"
    return "latency-or-underutilization-signal"


def render_markdown(summary):
    target = summary.get("target") or {}
    metrics = summary["metrics"]
    lines = [
        "# Targeted Nsight Compute Summary",
        "",
        f"Kernel: `{target.get('name', 'unknown')}`",
        "",
        f"Classification: **{summary['classification']}**.",
        "",
        "Collected sections: " + ", ".join(summary.get("sections") or ["unknown"]) + ".",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Duration | {metrics['Duration']:.2f} us |",
        f"| Compute (SM) throughput | {metrics['Compute (SM) Throughput']:.2f}% |",
        f"| Memory throughput | {metrics['Memory Throughput']:.2f}% |",
        f"| DRAM throughput | {metrics['DRAM Throughput']:.2f}% |",
        f"| L1/TEX throughput | {metrics['L1/TEX Cache Throughput']:.2f}% |",
        f"| L2 throughput | {metrics['L2 Cache Throughput']:.2f}% |",
    ]
    optional_rows = (
        ("Memory Throughput (Tbyte/s)", "Memory throughput", " TB/s"),
        ("L2 Hit Rate", "L2 hit rate", "%"),
        ("No Eligible", "Scheduler cycles with no eligible warp", "%"),
        ("Issued Warp Per Scheduler", "Issued warps per scheduler", ""),
        ("Theoretical Occupancy", "Theoretical occupancy", "%"),
        ("Achieved Occupancy", "Achieved occupancy", "%"),
        ("Long Scoreboard Stall Cycles", "Scoreboard-dependency stall", " cycles"),
        ("Long Scoreboard Stall Share", "Scoreboard-dependency share", "%"),
        ("Branch Efficiency", "Branch efficiency", "%"),
    )
    for key, label, suffix in optional_rows:
        if key in metrics:
            lines.append(f"| {label} | {metrics[key]:.2f}{suffix} |")
    lines.extend(
        [
            "",
            "This is a kernel- and launch-specific signal. Interpret memory, scheduler, "
            "instruction, and occupancy evidence together before claiming the full model "
            "is memory- or compute-bound.",
        ]
    )
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--details", type=Path, required=True)
    parser.add_argument("--target-json", type=Path)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    args = parser.parse_args()

    details = args.details.read_text(encoding="utf-8")
    metrics = parse_metrics(details)
    target = None
    if args.target_json and args.target_json.exists():
        target = json.loads(args.target_json.read_text(encoding="utf-8"))["selected"]
    summary = {
        "target": target,
        "sections": parse_sections(details),
        "metrics": metrics,
        "classification": classify(metrics),
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    args.markdown.write_text(render_markdown(summary), encoding="utf-8")


if __name__ == "__main__":
    main()
