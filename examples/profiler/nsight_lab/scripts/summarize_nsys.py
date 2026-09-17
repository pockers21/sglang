#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path


MODES = ("prefill-disabled", "decode-disabled", "decode-full")


def load_csv(path: Path, header_prefix: str, required=True):
    if not path.exists():
        if required:
            raise FileNotFoundError(path)
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    try:
        start = next(
            index for index, line in enumerate(lines) if line.startswith(header_prefix)
        )
    except StopIteration:
        if required:
            raise ValueError(f"Cannot find {header_prefix!r} header in {path}")
        return []
    return list(csv.DictReader(lines[start:]))


def summarize_mode(root: Path, mode: str):
    kernel_rows = load_csv(root / mode / "cuda_gpu_kern_sum.csv", "Time (%)")
    api_rows = load_csv(root / mode / "cuda_api_sum.csv", "Time (%)")
    memory_path = root / mode / "cuda_gpu_mem_time_sum.csv"
    memory_rows = load_csv(memory_path, "Time (%)", required=False)
    api = {row["Name"]: row for row in api_rows}

    ordinary_launches = {
        name: row
        for name, row in api.items()
        if name.startswith("cudaLaunchKernel") or name.startswith("cuLaunchKernel")
    }
    ordinary_calls = sum(int(row["Num Calls"]) for row in ordinary_launches.values())
    ordinary_time_ns = sum(
        int(row["Total Time (ns)"]) for row in ordinary_launches.values()
    )
    graph_calls = sum(
        int(row["Num Calls"])
        for name, row in api.items()
        if name.startswith("cudaGraphLaunch")
    )
    graph_time_ns = sum(
        int(row["Total Time (ns)"])
        for name, row in api.items()
        if name.startswith("cudaGraphLaunch")
    )

    return {
        "gpu_kernel_time_ms": sum(
            int(row["Total Time (ns)"]) for row in kernel_rows
        )
        / 1e6,
        "gpu_kernel_instances": sum(int(row["Instances"]) for row in kernel_rows),
        "gpu_memory_operation_time_ms": sum(
            int(row["Total Time (ns)"]) for row in memory_rows
        )
        / 1e6,
        "gpu_memory_operations": sum(
            int(
                row.get(
                    "Operations",
                    row.get("Instances", row.get("Count", 0)),
                )
            )
            for row in memory_rows
        ),
        "ordinary_launch_calls": ordinary_calls,
        "ordinary_launch_api_time_ms": ordinary_time_ns / 1e6,
        "graph_launch_calls": graph_calls,
        "graph_launch_api_time_ms": graph_time_ns / 1e6,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nsys-root", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args()

    summary = {mode: summarize_mode(args.nsys_root, mode) for mode in MODES}
    eager = summary["decode-disabled"]
    graph = summary["decode-full"]
    if eager["ordinary_launch_calls"] == 0:
        raise ValueError("The eager decode capture contains no CUDA kernel launches")
    summary["decode_comparison"] = {
        "gpu_kernel_time_change_percent": (
            graph["gpu_kernel_time_ms"] / eager["gpu_kernel_time_ms"] - 1.0
        )
        * 100.0,
        "ordinary_launch_call_reduction_percent": (
            1.0 - graph["ordinary_launch_calls"] / eager["ordinary_launch_calls"]
        )
        * 100.0,
    }

    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Nsight Systems Summary",
        "",
        "| Capture | GPU kernel time | Explicit GPU memory ops | Kernel instances | Ordinary launches | Graph launches |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for mode in MODES:
        data = summary[mode]
        lines.append(
            f"| `{mode}` | {data['gpu_kernel_time_ms']:.3f} ms | "
            f"{data['gpu_memory_operation_time_ms']:.3f} ms | {data['gpu_kernel_instances']} | "
            f"{data['ordinary_launch_calls']} | "
            f"{data['graph_launch_calls']} |"
        )
    comparison = summary["decode_comparison"]
    lines.extend(
        [
            "",
            "Decode GPU kernel time change with CUDA Graph: "
            f"**{comparison['gpu_kernel_time_change_percent']:+.2f}%**.",
            "",
            "Ordinary decode launch-call reduction with CUDA Graph: "
            f"**{comparison['ordinary_launch_call_reduction_percent']:.2f}%**.",
            "",
            "The CUDA Graph path still executes nearly the same GPU work. Its main "
            "benefit is replaying that work without one host launch per kernel.",
        ]
    )
    args.markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
