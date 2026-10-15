"""手动真实仿真验收：人为扰动执行器／场景，不伪造 VLM 或判分结果。

从项目根目录执行：PYTHONPATH=. python tests/validate_recovery.py --help。
"""

import argparse
import json
from pathlib import Path
from unittest.mock import patch

from agentic_manipulation import closed_loop as loop


def recovery_passed(result, fault):
    rounds = result["rounds"]
    if not result["injected_fault"]:
        return False
    if fault == "lost_target":
        return not result["success"] and result["state"] == "STOPPED" and (
            result["agent_status"] == "uncertain" or rounds[0].get("verdict", {}).get("recovery") == "stop"
            or "rejected" in result.get("failure", ""))
    expected = "replace" if fault == "failed_release" else "regrasp"
    recoveries = [r["recovery"] for r in rounds[1:]]
    return (result["success"] and result["agent_status"] == "success" and 2 <= len(rounds) <= 3
            and not rounds[0].get("truth", {}).get("success", True) and expected in recoveries
            and all(value in ("reobserve", expected) for value in recoveries))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--grasp-python", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--fault", action="append", choices=("missed_grasp", "failed_release", "lost_target"),
                        help="只复核指定故障；省略时检查全部三类")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    reports = []
    faults = args.fault or ["missed_grasp", "failed_release", "lost_target"]
    for fault in faults:
        original = loop.execute_place if fault == "failed_release" else loop.execute_pick
        injected = []

        def disturb(control, *positional, **keyword):
            if injected:
                return original(control, *positional, **keyword)
            injected.append(fault)
            step = control.step

            def broken_gripper(action):
                action = action.copy()
                action[-1] = -1 if fault == "failed_release" else 1
                return step(action)

            try:
                with patch.object(control, "step", side_effect=broken_gripper):
                    return original(control, *positional, **keyword)
            finally:
                if fault == "lost_target":
                    from mani_skill.utils.structs.pose import Pose
                    control.target_actor.set_pose(Pose.create_from_pq(p=[1.5, 1.5, 0.02]))

        name = "execute_place" if fault == "failed_release" else "execute_pick"
        with patch.object(loop, name, side_effect=disturb):
            result = loop.run_closed_loop(args.output / fault, args.grasp_python, args.checkpoint, 0, "把红色方块放进黄色碗里")
        result["injected_fault"] = injected
        rounds = result["rounds"]
        passed = recovery_passed(result, fault)
        result["recovery_check_passed"] = passed
        reports.append(result)
        (args.output / "summary.json").write_text(json.dumps({"faults": faults, "trials": reports, "passed": len(reports) == len(faults) and all(r["recovery_check_passed"] for r in reports)}, indent=2) + "\n", encoding="utf-8")
        print(f"{fault}: {'PASS' if passed else 'FAIL'}; rounds={len(rounds)}; {result.get('failure', result.get('stop_reason', ''))}", flush=True)
    assert all(r["recovery_check_passed"] for r in reports), "recovery acceptance failed; retain all reports"


if __name__ == "__main__":
    main()
