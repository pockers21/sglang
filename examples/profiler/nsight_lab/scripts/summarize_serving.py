#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


FIELDS = (
    "max_concurrency",
    "completed",
    "request_throughput",
    "output_throughput",
    "total_throughput",
    "p99_e2e_latency_ms",
    "p99_ttft_ms",
    "p99_tpot_ms",
)


def load_rows(path: Path):
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        missing = [field for field in FIELDS if field not in row]
        if missing:
            raise ValueError(f"{path}:{number} is missing {', '.join(missing)}")
        rows.append(row)
    if not rows:
        raise ValueError(f"No serving records found in {path}")
    return sorted(rows, key=lambda row: int(row["max_concurrency"]))


def summarize(rows, slo_ttft, slo_tpot):
    points = []
    for row in rows:
        point = {field: row[field] for field in FIELDS}
        point["max_concurrency"] = int(point["max_concurrency"])
        point["slo_pass"] = (
            float(point["p99_ttft_ms"]) <= slo_ttft
            and float(point["p99_tpot_ms"]) <= slo_tpot
        )
        points.append(point)

    best = max(points, key=lambda row: float(row["output_throughput"]))
    valid = [row for row in points if row["slo_pass"]]
    best_slo = max(valid, key=lambda row: float(row["output_throughput"])) if valid else None

    saturation = None
    for previous, current in zip(points, points[1:]):
        gain = float(current["output_throughput"]) / max(float(previous["output_throughput"]), 1e-9) - 1
        tpot_growth = float(current["p99_tpot_ms"]) / max(float(previous["p99_tpot_ms"]), 1e-9) - 1
        if gain < 0.10 and tpot_growth > 0.25:
            saturation = {
                "from_concurrency": previous["max_concurrency"],
                "at_concurrency": current["max_concurrency"],
                "output_throughput_gain_percent": gain * 100,
                "p99_tpot_growth_percent": tpot_growth * 100,
            }
            break

    return {
        "slo": {"p99_ttft_ms": slo_ttft, "p99_tpot_ms": slo_tpot},
        "points": points,
        "best_output_throughput": best,
        "best_slo_compliant": best_slo,
        "saturation_signal": saturation,
    }


def render_markdown(summary):
    lines = [
        "# Online Serving Sweep",
        "",
        f"SLO used for classification: p99 TTFT <= {summary['slo']['p99_ttft_ms']:.0f} ms and "
        f"p99 TPOT <= {summary['slo']['p99_tpot_ms']:.0f} ms.",
        "",
        "| Max concurrency | Completed | Requests/s | Output tok/s | Total tok/s | p99 TTFT | p99 TPOT | p99 E2E | SLO |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in summary["points"]:
        lines.append(
            f"| {row['max_concurrency']} | {row['completed']} | {float(row['request_throughput']):.2f} | "
            f"{float(row['output_throughput']):.1f} | {float(row['total_throughput']):.1f} | "
            f"{float(row['p99_ttft_ms']):.1f} ms | {float(row['p99_tpot_ms']):.1f} ms | "
            f"{float(row['p99_e2e_latency_ms']):.1f} ms | {'pass' if row['slo_pass'] else 'fail'} |"
        )
    best = summary["best_output_throughput"]
    lines.extend(
        [
            "",
            f"Peak measured output throughput: **{float(best['output_throughput']):.1f} tok/s** "
            f"at max concurrency **{best['max_concurrency']}**.",
        ]
    )
    if summary["best_slo_compliant"]:
        best_slo = summary["best_slo_compliant"]
        lines.extend(
            [
                "",
                f"Best SLO-compliant point: concurrency **{best_slo['max_concurrency']}**, "
                f"output throughput **{float(best_slo['output_throughput']):.1f} tok/s**.",
            ]
        )
    else:
        lines.extend(["", "No measured point satisfied both configured SLO thresholds."])
    if summary["saturation_signal"]:
        signal = summary["saturation_signal"]
        lines.extend(
            [
                "",
                f"Saturation signal begins near concurrency **{signal['at_concurrency']}**: output throughput "
                f"changed {signal['output_throughput_gain_percent']:+.1f}% while p99 TPOT changed "
                f"{signal['p99_tpot_growth_percent']:+.1f}% from the preceding point.",
            ]
        )
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    parser.add_argument("--slo-p99-ttft-ms", type=float, default=1000)
    parser.add_argument("--slo-p99-tpot-ms", type=float, default=100)
    args = parser.parse_args()

    summary = summarize(load_rows(args.input), args.slo_p99_ttft_ms, args.slo_p99_tpot_ms)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    args.markdown.write_text(render_markdown(summary), encoding="utf-8")


if __name__ == "__main__":
    main()
