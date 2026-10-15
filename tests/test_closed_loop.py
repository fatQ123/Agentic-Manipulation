"""闭环边界的一项离线检查；物理恢复由 validate_recovery 单独实测。"""

from contextlib import ExitStack
import copy
import tempfile
import time
from pathlib import Path
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from agentic_manipulation import closed_loop as loop


class ClosedLoopTests(unittest.TestCase):
    def test_fresh_replanning_limits_and_independent_truth(self):
        plan = {"status": "ready", "action": "pick_place", "object": {"color": "red", "bbox": [1, 2, 30, 40]},
                "destination": {"color": "yellow", "bbox": [50, 60, 90, 100]}, "steps": ["LOCATE_OBJECT", "GRASP", "PLACE"]}
        success = {"status": "success", "reason": "cube in bowl", "recovery": None}
        failed = {"status": "failure", "reason": "cube on table", "recovery": "regrasp"}
        uncertain = {"status": "uncertain", "reason": "occluded", "recovery": None}
        cases = [
            ([], [], False, 1, False),
            ([failed, success], [False, True], False, 2, True),
            ([failed] * 3, [False] * 3, False, 3, False),
            ([success], [False], False, 1, False),
            ([uncertain, uncertain], [False, False], False, 2, False),
            ([uncertain, success], [False, True], False, 2, True),
            ([dict(failed, recovery="stop")], [False], False, 1, False),
            ([dict(failed, recovery="replace"), success], [False, True], True, 2, True),
        ]
        cases = [(*case, None) for case in cases] + [
            ([success], [True], True, 1, True, "red"),
            ([], [], True, 1, False, "blue"),
        ]
        for verdicts, truths, held, count, expected, initial_held in cases:
            with self.subTest(verdicts=verdicts), tempfile.TemporaryDirectory() as root, ExitStack() as stack:
                mocks = {name: stack.enter_context(patch.object(loop, name)) for name in (
                    "configuration", "make_env", "PickController", "save_observation", "plan_task",
                    "infer_grasps", "execute_pick", "execute_place", "score_task", "verify", "reconstruct")}
                stack.enter_context(patch.dict("sys.modules", {"imageio": MagicMock(), "imageio.v2": MagicMock()}))
                mocks["configuration"].return_value = {"VLM_MODEL": "test"}
                control = mocks["PickController"].return_value
                control.pose.side_effect = lambda: np.eye(4)
                control.agent.is_grasping.return_value = held
                control.steps = 0
                if initial_held:
                    mocks["make_env"].return_value.unwrapped.cubes = {initial_held: MagicMock()}
                control.target_actor._objs[0].name = "cube"
                mocks["execute_pick"].return_value = (np.eye(4), np.zeros(3))
                if not verdicts:
                    mocks["execute_pick"].side_effect = RuntimeError("unexpected robot contact")
                plans = []
                for number in range(count):
                    fresh = copy.deepcopy(plan)
                    fresh["object"]["bbox"][0] += number
                    plans.append(fresh)
                mocks["plan_task"].side_effect = ValueError("place the same cube first") if initial_held == "blue" else plans
                mocks["verify"].side_effect = verdicts
                mocks["score_task"].side_effect = [{"success": value} for value in truths]
                stack.enter_context(patch.object(loop, "locate_destination", return_value=np.zeros(3)))
                cloud = MagicMock()
                cloud.__enter__.return_value = {"target_world": np.array([[0, 0, 0], [1, 1, 1]])}
                stack.enter_context(patch.object(loop.np, "load", return_value=cloud))
                report = loop.run_closed_loop(Path(root) / "run", "python", "weights", 0, "搬运")
                self.assertEqual(report["success"], expected)
                self.assertEqual(len(report["rounds"]), count)
                self.assertEqual(mocks["plan_task"].call_count, count)
                self.assertEqual(mocks["execute_pick"].call_count, 0 if initial_held else 1 if held or (verdicts and verdicts[0] == uncertain) else count)
                if not verdicts:
                    self.assertIn("same cube" if initial_held else "unexpected robot contact", report["failure"])
                    mocks["verify"].assert_not_called()
                for index, call in enumerate(mocks["plan_task"].call_args_list, 1):
                    self.assertIn(f"round-{index}/capture/rgb.png", str(call.args[1]))
                if verdicts == [success] and not expected:
                    self.assertEqual(report["vlm_judgments"]["false_success"], 1)
                    self.assertEqual(report["state"], "SUCCESS")
                if verdicts and verdicts[0] == uncertain:
                    self.assertEqual(report["rounds"][1]["recovery"], "reobserve")
                    self.assertEqual(mocks["infer_grasps"].call_count, 1)
                    self.assertEqual(report["state"], "SUCCESS" if expected else "STOPPED")
                self.assertTrue((Path(root) / "run/result.json").is_file())
        with patch.object(loop.time, "monotonic", return_value=10):
            with self.assertRaises(TimeoutError):
                loop.ResourceManager(10).remaining(60)
            self.assertEqual(loop.ResourceManager(12).remaining(60), 2)
        with tempfile.TemporaryDirectory() as root, patch.dict("sys.modules", {"imageio": MagicMock(), "imageio.v2": MagicMock()}), patch.object(loop, "TASK_TIMEOUT", 0.02), patch.object(loop, "configuration", side_effect=lambda: time.sleep(1)):
            report = loop.run_closed_loop(Path(root) / "timeout", "python", "weights", 0, "搬运")
            self.assertFalse(report["success"])
            self.assertEqual(report["failure"], "task deadline exceeded")
            self.assertLess(report["elapsed_s"], 0.5)
        trial = {}
        request = MagicMock(side_effect=[ConnectionError("temporary disconnect"), success])
        self.assertEqual(loop.ResourceManager(time.monotonic() + 5).call_vlm(request, "current.png", trial=trial, stage="verification", retry=True), success)
        self.assertEqual(trial["verification_requests"], 2)
        self.assertTrue(all(call.args == ("current.png",) for call in request.call_args_list))
        for error, calls in ((ConnectionError("offline"), 2), (ValueError("malformed"), 1), (RuntimeError("HTTP 401"), 1)):
            request = MagicMock(side_effect=error)
            with self.assertRaises(type(error)):
                loop.ResourceManager(time.monotonic() + 5).call_vlm(request, trial={}, stage="plan", retry=True)
            self.assertEqual(request.call_count, calls)
        request = MagicMock()
        with self.assertRaises(TimeoutError):
            loop.ResourceManager(time.monotonic() - 1).call_vlm(request, trial={}, stage="plan", retry=True)
        request.assert_not_called()
        changed = copy.deepcopy(plan)
        changed["object"]["color"] = "blue"
        self.assertNotEqual(loop.task_identity(plan), loop.task_identity(changed))


if __name__ == "__main__":
    unittest.main()
