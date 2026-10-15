"""V1／V2／V3 固定种子 0–9 验收，保留所有成功与失败。"""

import argparse
import json
import os
import urllib.parse
from pathlib import Path
import subprocess
import sys


V2_CASES = [
    ("把红色方块放进黄色碗里", "red", "yellow"),
    ("把蓝色方块放进紫色碗里", "blue", "purple"),
    ("把绿色方块放进黄色碗里", "green", "yellow"),
    ("把红色方块放进紫色碗里", "red", "purple"),
    ("把蓝色方块放进黄色碗里", "blue", "yellow"),
    ("把绿色方块放进紫色碗里", "green", "purple"),
    ("将红色积木移入黄色容器", "red", "yellow"),
    ("将蓝色积木移入紫色容器", "blue", "purple"),
    ("将绿色积木移入黄色容器", "green", "yellow"),
    ("将红色积木移入紫色容器", "red", "purple"),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--grasp-python", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    versions = parser.add_mutually_exclusive_group()
    versions.add_argument("--v2", action="store_true", help="固定十组自然语言搬运验收")
    versions.add_argument("--v3", action="store_true", help="同十组场景的闭环验收与 VLM 误判统计")
    args = parser.parse_args()
    language = args.v2 or args.v3
    version = "v3" if args.v3 else "v2" if args.v2 else "v1"
    threshold = 8
    if args.v3:
        baseline = json.loads((Path(__file__).resolve().parents[1] / "docs/validation/v2-results.json").read_text(encoding="utf-8"))
        if baseline["total"] != 10 or len(baseline["trials"]) != 10:
            parser.error("V3 requires the complete ten-case V2 baseline")
        threshold = max(9, baseline["successes"])
    environment = None
    service = {}
    if language:
        from .vlm import configuration
        try:
            config = configuration()
        except (OSError, ValueError) as exc:
            parser.exit(1, f"VLM 配置错误：{exc}\n")
        environment = os.environ | config
        service = {"model": config["VLM_MODEL"], "service_host": urllib.parse.urlsplit(config["VLM_BASE_URL"]).hostname}
    try:
        args.output.mkdir(parents=True, exist_ok=False)
    except OSError as exc:
        parser.exit(1, f"验收目录创建失败：{exc}\n")
    results = []
    for seed in range(10):
        directory = (args.output / f"seed-{seed}").resolve()
        with (args.output / f"seed-{seed}.log").open("w", encoding="utf-8") as log:
            try:
                command = [
                    sys.executable, "-m", "agentic_manipulation.pick", "--seed", str(seed),
                    "--output", str(directory), "--grasp-python", str(args.grasp_python.resolve()),
                    "--checkpoint", str(args.checkpoint.resolve()),
                ]
                if language:
                    command.extend(["--instruction", V2_CASES[seed][0]])
                if args.v3:
                    command.append("--v3")
                process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=240, env=environment)
                result = json.loads((directory / "result.json").read_text(encoding="utf-8"))
                result["success"] = result["success"] is True and process.returncode == 0
                if language:
                    _, expected_object, expected_bowl = V2_CASES[seed]
                    plan = result.get("plan", {})
                    correct = plan.get("action") == "pick_place" and plan.get("object", {}).get("color") == expected_object and (plan.get("destination") or {}).get("color") == expected_bowl
                    result["expected"] = {"object": expected_object, "destination": expected_bowl}
                    result["success"] &= correct
                    if args.v3:
                        result["success"] &= result.get("agent_status") == "success"
                    if not correct:
                        result["failure"] = result.get("failure", "wrong language target")
            except (subprocess.TimeoutExpired, OSError, ValueError, KeyError) as exc:
                result = {"seed": seed, "success": False, "failure": str(exc)}
        results.append(result)
        summary = {**service, "version": version, "trials": results, "successes": sum(r["success"] for r in results), "total": len(results)}
        summary["required_successes"] = threshold
        summary["passed"] = len(results) == 10 and summary["successes"] >= threshold
        if args.v3:
            summary["v2_baseline_successes"] = baseline["successes"]
            summary["vlm_judgments"] = {key: sum(r.get("vlm_judgments", {}).get(key, 0) for r in results)
                                        for key in ("total", "false_success", "false_failure", "uncertain")}
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(f"seed {seed}: {'PASS' if result['success'] else 'FAIL'} {result.get('failure', '')}", flush=True)
    assert len(results) == 10 and {r["seed"] for r in results} == set(range(10))
    print(f"{version.upper()} acceptance: {summary['successes']}/10", flush=True)
    if not summary["passed"]:
        parser.exit(1)


if __name__ == "__main__":
    main()
