"""新增三个必要检查：选择器、资源回收、持续会话。"""

import json
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from agentic_manipulation.grasp_selector import ExecutionFailure, select_grasp
from agentic_manipulation.resource_manager import ResourceManager
from agentic_manipulation.demo import DemoSession


class ModuleTests(unittest.TestCase):
    def test_selector_uses_executable_candidate_and_shared_placement(self):
        rotation = np.array([[0, 1, 0], [0, 0, -1], [-1, 0, 0]])
        grasp = np.r_[0.9, .06, .02, .03, rotation.ravel(), [0, 0, .06], -1]
        candidates = np.stack([grasp, grasp.copy()])
        candidates[1, 0] = .8
        candidates[1, 15] = .07
        points = np.array([[-.02, -.02, .01], [.02, .02, .08]])
        reached = []

        def reachable(poses):
            reached.append(poses)
            return poses[3][2, 3] > .045

        counts = {}
        result = select_grasp(candidates, points, np.eye(4), reachable, np.array([.1, .1, .09]), counts)
        self.assertEqual(result["index"], 1)
        self.assertEqual(counts["ik"], 1)
        self.assertEqual(len(reached[-1]), 11)
        for checked, planned in zip(reached[-1][5:], result["placement"].values()):
            self.assertIs(checked, planned)
        observed = select_grasp(candidates, points, np.eye(4), lambda poses: True, observe_held=True)
        np.testing.assert_allclose(observed["observation"][:3, 2], [0, 0, -1])
        self.assertEqual(observed["observation"][2, 3], .25)
        with self.assertRaises(ExecutionFailure):
            select_grasp(candidates, points, np.eye(4), lambda poses: False)
        with self.assertRaisesRegex(ValueError, "invalid"):
            select_grasp(np.full((1, 17), np.nan), points, np.eye(4), reachable)

    def test_resource_admission_failures_timeout_and_cancel_reap_child(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            with patch.object(ResourceManager, "gpu_memory", return_value={"total_mib": 6144, "free_mib": 1}), patch("agentic_manipulation.resource_manager.subprocess.Popen") as popen:
                with self.assertRaisesRegex(RuntimeError, "insufficient"):
                    ResourceManager().run_grasp([sys.executable], root / "low.log", root)
                popen.assert_not_called()
            with patch.object(ResourceManager, "_query", side_effect=RuntimeError("monitor unavailable")):
                with self.assertRaisesRegex(RuntimeError, "monitor"):
                    ResourceManager().run_grasp([sys.executable], root / "monitor.log", root)
            for mode in ("success", "failure", "oom", "timeout", "cancel", "unexpected"):
                with self.subTest(mode=mode), patch.object(ResourceManager, "gpu_memory", return_value={"total_mib": 6144, "free_mib": 4000}):
                    resources = ResourceManager(time.monotonic() + (0.15 if mode == "timeout" else 5))
                    program = "import time; time.sleep(10)" if mode in ("timeout", "cancel", "unexpected") else "raise SystemExit(" + ("3" if mode == "failure" else "0") + ")"
                    if mode == "oom":
                        program = "import sys; print('CUDA out of memory',file=sys.stderr); sys.exit(1)"
                    sample = patch.object(resources, "process_memory", side_effect=KeyboardInterrupt if mode == "cancel" else IndexError if mode == "unexpected" else None, return_value=100)
                    with sample:
                        if mode == "success":
                            with resources:
                                resources.run_grasp([sys.executable, "-c", program], root / (mode + ".log"), root)
                        else:
                            error = KeyboardInterrupt if mode == "cancel" else TimeoutError if mode == "timeout" else IndexError if mode == "unexpected" else RuntimeError
                            with self.assertRaisesRegex(error, "CUDA OOM" if mode == "oom" else ""), resources:
                                resources.run_grasp([sys.executable, "-c", program], root / (mode + ".log"), root)
                    self.assertIsNone(resources.process)
                    pid = resources.report["events"][0]["pid"]
                    with self.assertRaises(ProcessLookupError):
                        os.kill(pid, 0)
                    self.assertIsNone(resources.busy)
            resources = ResourceManager()
            with resources.phase("execute"):
                with self.assertRaisesRegex(RuntimeError, "already active"):
                    with resources.phase("grasp_inference"):
                        pass

    def test_session_reuses_scene_and_stops_after_motion_failure(self):
        env = MagicMock()
        env.unwrapped = env
        env.agent.tcp.pose.to_transformation_matrix.return_value.__getitem__.return_value.cpu.return_value.numpy.return_value = np.eye(4)
        reports = [
            {"success": True, "agent_status": "success", "steps": 400, "elapsed_s": 1, "state": "SUCCESS", "held_object": None},
            {"success": False, "agent_status": "stopped", "steps": 0, "elapsed_s": 1, "state": "IDLE", "held_object": None},
            {"success": True, "agent_status": "success", "steps": 200, "elapsed_s": 1, "state": "SUCCESS", "held_object": "blue"},
            {"success": False, "agent_status": "uncertain", "steps": 300, "elapsed_s": 1, "state": "STOPPED", "held_object": None},
        ]
        with tempfile.TemporaryDirectory() as root, patch("agentic_manipulation.demo.run_closed_loop", side_effect=reports) as run:
            session = DemoSession(env, Path(root) / "session", "python", "weights", seed=0)
            for command in ("搬运红色", "不明确指令", "搬运蓝色", "搬运绿色"):
                session.submit(command)
            self.assertEqual(env.reset.call_count, 1)
            self.assertTrue(all(call.kwargs["env"] is env for call in run.call_args_list))
            self.assertEqual([call.args[0].name for call in run.call_args_list], ["task-001", "task-002", "task-003", "task-004"])
            self.assertTrue(session.blocked)
            self.assertIn("reset", session.submit("下一条")["failure"])
            self.assertEqual(run.call_count, 4)
            session.reset()
            self.assertEqual(env.reset.call_count, 2)
            self.assertFalse(session.blocked)
            saved = json.loads((session.output / "session.json").read_text())
            self.assertEqual(len(saved["tasks"]), 4)
            self.assertEqual(session.tasks_since_reset, 0)
            run.side_effect = [{"success": False, "agent_status": "stopped", "steps": 0, "elapsed_s": 1, "state": "STOPPED", "held_object": None}]
            session.submit("资源不足")
            self.assertTrue(session.blocked)
            session.reset()
            run.side_effect = KeyboardInterrupt
            with self.assertRaises(KeyboardInterrupt):
                session.submit("取消本条任务")
            self.assertEqual(session.events[-1]["command"], "interrupted")
            self.assertTrue(session.blocked)
            session.close()
            env.close.assert_called_once()
