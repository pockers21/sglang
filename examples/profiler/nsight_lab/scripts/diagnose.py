#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def read_optional(path):
    if path is None or not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def finding(name, status, evidence, next_step):
    return {"name": name, "status": status, "evidence": evidence, "next_step": next_step}


def build_diagnosis(ab, nsys, offline, serving, ncu):
    findings = []
    if ab and nsys:
        speedup = float(ab["decode_speedup"])
        comparison = nsys["decode_comparison"]
        gpu_change = float(comparison["gpu_kernel_time_change_percent"])
        launch_reduction = float(comparison["ordinary_launch_call_reduction_percent"])
        supported = speedup >= 1.15 and abs(gpu_change) <= 10 and launch_reduction >= 50
        findings.append(
            finding(
                "Host launch overhead",
                "supported" if supported else "not-supported",
                f"CUDA Graph speedup={speedup:.3f}x, GPU kernel time change={gpu_change:+.2f}%, "
                f"ordinary launch reduction={launch_reduction:.2f}%.",
                "Keep CUDA Graph enabled for matching steady-state shapes."
                if supported
                else "Inspect the paired timeline before attributing the result to launch overhead.",
            )
        )
    else:
        findings.append(
            finding(
                "Host launch overhead",
                "unknown",
                "CUDA Graph A/B or paired Nsight Systems summary is missing.",
                "Run the A/B and paired decode captures.",
            )
        )

    if offline:
        best = offline["best_overall_throughput"]
        findings.append(
            finding(
                "Offline shape sensitivity",
                "measured",
                f"Best measured point is batch={best['batch_size']}, input={best['input_len']}, "
                f"output={best['output_len']} at {best['overall_throughput']:.1f} tok/s.",
                "Compare the production request shape to the full sweep instead of quoting one batch-1 number.",
            )
        )
    else:
        findings.append(
            finding("Offline shape sensitivity", "unknown", "No shape sweep is present.", "Run run_offline_sweep.sh.")
        )

    if serving:
        best = serving["best_output_throughput"]
        slo = serving.get("best_slo_compliant")
        evidence = (
            f"Peak output throughput={float(best['output_throughput']):.1f} tok/s at concurrency "
            f"{best['max_concurrency']}."
        )
        if slo:
            evidence += (
                f" Best SLO-compliant concurrency={slo['max_concurrency']} at "
                f"{float(slo['output_throughput']):.1f} tok/s."
            )
        findings.append(
            finding(
                "Serving saturation",
                "measured",
                evidence,
                "Use the SLO-compliant point for capacity planning; peak throughput alone is not a serving target.",
            )
        )
    else:
        findings.append(
            finding("Serving saturation", "unknown", "No online concurrency sweep is present.", "Run run_serving_sweep.sh.")
        )

    if ncu:
        if "metrics" in ncu:
            selected = ncu.get("target") or {}
            metrics = ncu["metrics"]
            evidence = (
                f"Kernel {selected.get('name', 'unknown')} measured memory throughput "
                f"{metrics['Memory Throughput']:.2f}% and compute throughput "
                f"{metrics['Compute (SM) Throughput']:.2f}% in the selected launch."
            )
            next_step = (
                "Inspect the full NCU memory, scheduler, instruction, and occupancy sections before "
                "generalizing this kernel-level signal to the model."
            )
            if "Long Scoreboard Stall Share" in metrics:
                evidence += (
                    f" Scheduler cycles with no eligible warp={metrics['No Eligible']:.2f}%, "
                    f"achieved occupancy={metrics['Achieved Occupancy']:.2f}%, and "
                    f"scoreboard-dependency share={metrics['Long Scoreboard Stall Share']:.2f}%."
                )
                next_step = (
                    "Map the long-scoreboard stalls back to source counters and memory access patterns, "
                    "then validate any kernel change with end-to-end A/B timing."
                )
            findings.append(
                finding(
                    "Kernel deep dive",
                    ncu["classification"],
                    evidence,
                    next_step,
                )
            )
        else:
            selected = ncu["selected"]
            findings.append(
                finding(
                    "Kernel deep dive",
                    "target-selected",
                    f"Selected {selected['name']} from {selected['time_percent']:.2f}% of captured GPU kernel time.",
                    "Run or parse the targeted NCU report to classify this launch.",
                )
            )
    else:
        findings.append(
            finding(
                "Compute versus memory bottleneck",
                "unknown",
                "Nsight Systems can rank kernels but cannot establish a kernel's compute or memory limit.",
                "Run one targeted Nsight Compute capture before making a roofline-style claim.",
            )
        )
    return {"findings": findings}


def render_markdown(result):
    lines = ["# Evidence-Based Diagnosis", ""]
    for item in result["findings"]:
        lines.extend(
            [
                f"## {item['name']}: {item['status']}",
                "",
                item["evidence"],
                "",
                f"Next step: {item['next_step']}",
                "",
            ]
        )
    lines.extend(
        [
            "## Claim boundary",
            "",
            "Every conclusion above is scoped to the recorded model, SGLang revision, workload, GPU, and runtime stack. "
            "A missing profiler layer is reported as unknown rather than guessed.",
        ]
    )
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ab-json", type=Path)
    parser.add_argument("--nsys-json", type=Path)
    parser.add_argument("--offline-json", type=Path)
    parser.add_argument("--serving-json", type=Path)
    parser.add_argument("--ncu-target-json", type=Path)
    parser.add_argument("--ncu-summary-json", type=Path)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    args = parser.parse_args()

    result = build_diagnosis(
        read_optional(args.ab_json),
        read_optional(args.nsys_json),
        read_optional(args.offline_json),
        read_optional(args.serving_json),
        read_optional(args.ncu_summary_json) or read_optional(args.ncu_target_json),
    )
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    args.markdown.write_text(render_markdown(result), encoding="utf-8")


if __name__ == "__main__":
    main()
