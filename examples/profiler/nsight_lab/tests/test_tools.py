import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


LAB_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = LAB_ROOT / "scripts"


def run_script(name, *args):
    subprocess.run(
        [sys.executable, str(SCRIPTS / name), *map(str, args)],
        check=True,
        capture_output=True,
        text=True,
    )


class ToolTests(unittest.TestCase):
    def test_ncu_profile_derives_capacity_from_shape(self):
        with tempfile.TemporaryDirectory() as directory:
            environment = os.environ.copy()
            environment.update(
                {
                    "NCU": "/bin/echo",
                    "OUTPUT_ROOT": directory,
                    "INPUT_LEN": "256",
                    "OUTPUT_LEN": "4",
                    "BATCH_SIZE": "2",
                }
            )
            result = subprocess.run(
                ["bash", str(SCRIPTS / "profile_kernel_ncu.sh"), "/tmp/model"],
                check=True,
                capture_output=True,
                text=True,
                env=environment,
            )
            self.assertIn("--context-length 260", result.stdout)
            self.assertIn("--max-total-tokens 520", result.stdout)
            self.assertIn("--section SchedulerStats", result.stdout)

    def test_ncu_profile_rejects_explicit_small_capacity(self):
        with tempfile.TemporaryDirectory() as directory:
            environment = os.environ.copy()
            environment.update(
                {
                    "NCU": "/usr/bin/true",
                    "OUTPUT_ROOT": directory,
                    "INPUT_LEN": "256",
                    "OUTPUT_LEN": "4",
                    "CONTEXT_LENGTH": "128",
                }
            )
            result = subprocess.run(
                ["bash", str(SCRIPTS / "profile_kernel_ncu.sh"), "/tmp/model"],
                check=False,
                capture_output=True,
                text=True,
                env=environment,
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn("need at least 260", result.stderr)

    def test_ncu_summary_classifies_memory_signal(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            details = root / "details.txt"
            details.write_text(
                """Section: GPU Speed Of Light Throughput
Metric Name             Metric Unit Metric Value
Duration                         us        52.96
Memory Throughput                 %        78.51
DRAM Throughput                   %        78.51
L1/TEX Cache Throughput           %        22.22
L2 Cache Throughput               %        84.32
Compute (SM) Throughput           %         7.43
Section: Memory Workload Analysis
Memory Throughput                Tbyte/s     3.88
L2 Hit Rate                            %     2.14
Section: Scheduler Statistics
No Eligible                            %    91.28
Issued Warp Per Scheduler                    0.09
Section: Occupancy
Theoretical Occupancy                  %    18.75
Achieved Occupancy                     %    13.85
On average, each warp spends 21.5 cycles being stalled waiting for a scoreboard dependency.
This stall type represents about 84.8% of the issue interval.
Section: Source Counters
Branch Efficiency                      %    99.59
"""
            )
            output = root / "summary.json"
            markdown = root / "summary.md"
            run_script(
                "summarize_ncu.py",
                "--details", details,
                "--json", output,
                "--markdown", markdown,
            )
            result = json.loads(output.read_text())
            self.assertEqual(result["classification"], "memory-throughput-dominant")
            self.assertEqual(
                result["sections"],
                [
                    "GPU Speed Of Light Throughput",
                    "Memory Workload Analysis",
                    "Scheduler Statistics",
                    "Occupancy",
                    "Source Counters",
                ],
            )
            self.assertEqual(result["metrics"]["Memory Throughput"], 78.51)
            self.assertEqual(result["metrics"]["Memory Throughput (Tbyte/s)"], 3.88)
            self.assertEqual(result["metrics"]["L2 Cache Throughput"], 84.32)
            self.assertEqual(result["metrics"]["No Eligible"], 91.28)
            self.assertEqual(result["metrics"]["Achieved Occupancy"], 13.85)
            self.assertEqual(result["metrics"]["Long Scoreboard Stall Share"], 84.8)
            self.assertEqual(result["metrics"]["Branch Efficiency"], 99.59)

    def test_nsys_summary_allows_missing_memory_csv(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for mode in ("prefill-disabled", "decode-disabled", "decode-full"):
                capture = root / mode
                capture.mkdir()
                capture.joinpath("cuda_gpu_kern_sum.csv").write_text(
                    'Time (%),Total Time (ns),Instances,Avg (ns),Med (ns),Min (ns),Max (ns),StdDev (ns),Name\n'
                    '100.0,1000000,10,1,1,1,1,0,kernel\n'
                )
                launch_calls = 100 if mode != "decode-full" else 2
                capture.joinpath("cuda_api_sum.csv").write_text(
                    'Time (%),Total Time (ns),Num Calls,Avg (ns),Med (ns),Min (ns),Max (ns),StdDev (ns),Name\n'
                    f'100.0,1000000,{launch_calls},1,1,1,1,0,cudaLaunchKernel\n'
                )
            root.joinpath("decode-disabled", "cuda_gpu_mem_time_sum.csv").write_text(
                'Time (%),Total Time (ns),Count,Avg (ns),Med (ns),Min (ns),Max (ns),StdDev (ns),Operation\n'
                '100.0,32000,2,16000,16000,16000,16000,0,[CUDA memset]\n'
            )
            output = root / "summary.json"
            markdown = root / "summary.md"
            run_script(
                "summarize_nsys.py",
                "--nsys-root", root,
                "--json", output,
                "--markdown", markdown,
            )
            result = json.loads(output.read_text())
            self.assertEqual(result["decode-disabled"]["gpu_memory_operation_time_ms"], 0.032)
            self.assertEqual(result["decode-disabled"]["gpu_memory_operations"], 2)
            self.assertEqual(result["decode-full"]["gpu_memory_operations"], 0)
            self.assertEqual(result["decode_comparison"]["ordinary_launch_call_reduction_percent"], 98)

    def test_offline_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input.jsonl"
            rows = [
                {
                    "batch_size": 1,
                    "input_len": 128,
                    "output_len": 32,
                    "prefill_latency": 0.01,
                    "prefill_throughput": 12800,
                    "median_decode_latency": 0.005,
                    "median_decode_throughput": 200,
                    "total_latency": 0.16,
                    "overall_throughput": 1000,
                },
                {
                    "batch_size": 4,
                    "input_len": 128,
                    "output_len": 32,
                    "prefill_latency": 0.02,
                    "prefill_throughput": 25600,
                    "median_decode_latency": 0.007,
                    "median_decode_throughput": 571.4,
                    "total_latency": 0.22,
                    "overall_throughput": 2900,
                },
            ]
            source.write_text("".join(json.dumps(row) + "\n" for row in rows))
            output = root / "summary.json"
            markdown = root / "summary.md"
            run_script("summarize_offline_sweep.py", "--input", source, "--json", output, "--markdown", markdown)
            summary = json.loads(output.read_text())
            self.assertEqual(summary["best_overall_throughput"]["batch_size"], 4)
            self.assertIn("2.900", f"{summary['best_overall_throughput']['overall_throughput'] / 1000:.3f}")

    def test_serving_summary_and_slo(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input.jsonl"
            rows = []
            for concurrency, throughput, tpot in [(1, 100, 10), (8, 500, 20), (32, 520, 40)]:
                rows.append(
                    {
                        "max_concurrency": concurrency,
                        "completed": 64,
                        "request_throughput": throughput / 64,
                        "output_throughput": throughput,
                        "total_throughput": throughput * 5,
                        "p99_e2e_latency_ms": tpot * 70,
                        "p99_ttft_ms": 200,
                        "p99_tpot_ms": tpot,
                    }
                )
            source.write_text("".join(json.dumps(row) + "\n" for row in rows))
            output = root / "summary.json"
            markdown = root / "summary.md"
            run_script(
                "summarize_serving.py",
                "--input", source,
                "--json", output,
                "--markdown", markdown,
                "--slo-p99-ttft-ms", "500",
                "--slo-p99-tpot-ms", "25",
            )
            summary = json.loads(output.read_text())
            self.assertEqual(summary["best_output_throughput"]["max_concurrency"], 32)
            self.assertEqual(summary["best_slo_compliant"]["max_concurrency"], 8)
            self.assertEqual(summary["saturation_signal"]["at_concurrency"], 32)

    def test_ncu_target_selects_hottest_kernel(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "kernels.csv"
            source.write_text(
                "Generating report...\n"
                'Time (%),Total Time (ns),Instances,Avg (ns),Med (ns),Min (ns),Max (ns),StdDev (ns),Name\n'
                '20.0,2000000,20,1,1,1,1,0,cold_kernel(int)\n'
                '80.0,8000000,10,1,1,1,1,0,hot_kernel<float>(float*)\n'
            )
            output = root / "target.json"
            run_script("select_ncu_target.py", "--input", source, "--json", output)
            selected = json.loads(output.read_text())["selected"]
            self.assertEqual(selected["name"], "hot_kernel<float>(float*)")
            self.assertTrue(selected["ncu_kernel_filter"].startswith("regex:^"))

    def test_diagnosis_requires_matching_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ab = root / "ab.json"
            nsys = root / "nsys.json"
            ab.write_text(json.dumps({"decode_speedup": 2.0}))
            nsys.write_text(
                json.dumps(
                    {
                        "decode_comparison": {
                            "gpu_kernel_time_change_percent": 2.5,
                            "ordinary_launch_call_reduction_percent": 98.0,
                        }
                    }
                )
            )
            output = root / "diagnosis.json"
            markdown = root / "diagnosis.md"
            run_script(
                "diagnose.py",
                "--ab-json", ab,
                "--nsys-json", nsys,
                "--json", output,
                "--markdown", markdown,
            )
            result = json.loads(output.read_text())
            self.assertEqual(result["findings"][0]["status"], "supported")
            self.assertEqual(result["findings"][-1]["status"], "unknown")

    def test_diagnosis_uses_extended_ncu_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ncu = root / "ncu.json"
            ncu.write_text(
                json.dumps(
                    {
                        "target": {"name": "hot_kernel"},
                        "classification": "memory-throughput-dominant",
                        "metrics": {
                            "Memory Throughput": 78.0,
                            "Compute (SM) Throughput": 7.0,
                            "No Eligible": 91.0,
                            "Achieved Occupancy": 14.0,
                            "Long Scoreboard Stall Share": 84.0,
                        },
                    }
                )
            )
            output = root / "diagnosis.json"
            markdown = root / "diagnosis.md"
            run_script(
                "diagnose.py",
                "--ncu-summary-json", ncu,
                "--json", output,
                "--markdown", markdown,
            )
            finding = json.loads(output.read_text())["findings"][-1]
            self.assertIn("scoreboard-dependency share=84.00%", finding["evidence"])
            self.assertIn("source counters", finding["next_step"])


if __name__ == "__main__":
    unittest.main()
