"""无需 GPU/API 的三栏投影和终端事件检查。"""

from contextlib import redirect_stdout
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from agentic_manipulation.debug_view import DebugWindow, GraspDebugView, project_grasp, selected_grasp_data
from agentic_manipulation.progress import TerminalProgress


class DebugViewTests(unittest.TestCase):
    def test_projection_selection_and_window_lifecycle(self):
        grasp = np.r_[.9, .06, .02, .03, np.eye(3).ravel(), [0, 0, 1], -1]
        intrinsic = np.array([[100, 0, 160], [0, 100, 120], [0, 0, 1]])
        extrinsic = np.eye(4)
        extrinsic[0, 3] = .1
        pixels = project_grasp(grasp, intrinsic, extrinsic)
        np.testing.assert_allclose(pixels, [[173, 117], [168, 117], [168, 123], [173, 123]])
        behind = grasp.copy()
        behind[15] = -1
        self.assertIsNone(project_grasp(behind, intrinsic, extrinsic))
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            (root / "capture").mkdir()
            (root / "grasps").mkdir()
            Image.fromarray(np.zeros((240, 320, 3), dtype=np.uint8)).save(root / "capture/rgb.png")
            np.save(root / "capture/depth.npy", np.ones((240, 320)))
            np.save(root / "capture/camera_intrinsic.npy", intrinsic)
            np.save(root / "capture/camera_extrinsic.npy", extrinsic)
            np.save(root / "grasps/grasps_world.npy", np.stack([behind, grasp]))
            data = selected_grasp_data({"directory": root, "target": np.eye(4), "report": {"candidate_index": 1}})
            np.testing.assert_allclose(data["camera_position"], [.1, 0, 1])
            np.testing.assert_array_equal(data["grasp"], grasp)
            with patch.object(plt, "show"):
                view = GraspDebugView()
            try:
                view("observation", {"directory": root})
                view("plan_ready", {"plan": {"object": {"color": "blue", "bbox": [100, 200, 300, 400]},
                                              "destination": {"color": "yellow", "bbox": [600, 500, 900, 800]}}})
                np.testing.assert_allclose(view.axes[0].patches[0].get_bbox().bounds, [32, 48, 64, 48])
                view("selected_grasp", data)
                self.assertTrue((root / "grasp-debug.png").is_file())
                self.assertEqual(len(view.axes[2].lines), 1)
                view("before_step", {})
                self.assertEqual(len(view.axes[2].lines), 1)
                view("observation", {"directory": root})
                self.assertFalse(view.annotations)
                self.assertEqual(len(view.axes[2].lines), 0)
            finally:
                view.close()
            self.assertFalse(plt.fignum_exists(view.fig.number))
            view("instruction", {"instruction": "关闭额外窗口后继续"})
            (root / "grasp-debug.png").unlink()
            window = DebugWindow()
            try:
                window("observation", {"directory": root})
                window("selected_grasp", data)
            finally:
                window.close()
            self.assertEqual(window.process.exitcode, 0)
            self.assertTrue((root / "grasp-debug.png").is_file())
            window("instruction", {"instruction": "子进程退出后继续"})

    def test_terminal_stages_retry_and_no_frame_spam(self):
        output = io.StringIO()
        progress = TerminalProgress()
        with redirect_stdout(output):
            progress("instruction", {"instruction": "把蓝色方块放进黄色碗"})
            progress("observe", {"round": 1})
            progress("plan", {})
            progress("plan_ready", {"plan": {"object": {"color": "blue", "bbox": [1, 2, 3, 4]},
                                              "destination": {"color": "yellow", "bbox": [5, 6, 7, 8]}}})
            progress("grasp_inference", {})
            progress("selected_grasp", {"camera_position": np.array([.1, .2, .3]), "target": np.eye(4),
                                        "grasp": np.array([.9, .06])})
            progress("raise", {})
            progress("place", {})
            progress("state", {"state": "RECOVERING", "reason": "cube on table"})
            progress("observe", {"round": 2})
            progress("locate", {})
            before = output.getvalue()
            for _ in range(10):
                progress("frame", {})
                progress("before_step", {})
            self.assertEqual(output.getvalue(), before)
            progress("task_complete", {"report": {"state": "STOPPED", "success": False, "failure": "timeout"}})
            progress("instruction", {"instruction": "下一条"})
            self.assertEqual(progress.round, 1)
        text = output.getvalue()
        for expected in ("[场景]", "planned 1 task", "GraspNet for blue_cube", "camera position [0.1, 0.2, 0.3]",
                         "atomic pick", "atomic place into yellow_bowl", "[重试] cube on table",
                         "grounding attempt 2", "STOPPED; success=False; timeout"):
            self.assertIn(expected, text)
