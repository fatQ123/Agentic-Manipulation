"""V2 仅保留协议拒绝执行与视觉框点云两个必要检查。"""

import copy
import http.client
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, MagicMock
import urllib.error

import numpy as np
from PIL import Image

from agentic_manipulation.geometry import grounded_region
from agentic_manipulation.vlm import plan as backend_plan, locate, validate_plan, verify, validate_verdict
from agentic_manipulation.task_planner import plan_task, task_identity


class LanguageTests(unittest.TestCase):
    def test_service_response_and_fail_closed(self):
        plan = {"status": "ready", "action": "pick_place", "object": {"color": "red", "bbox": [10, 20, 50, 60]}, "destination": {"color": "yellow", "bbox": [100, 200, 400, 600]}}
        self.assertEqual(validate_plan(plan), plan)
        for field, value in (("status", "unknown"), ("action", "delete"), ("destination", None), ("object", {"color": "red", "bbox": [-1, 0, 10, 10]}), ("object", {"color": "red", "bbox": [0, 0, True, 1]})):
            broken = copy.deepcopy(plan)
            broken[field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                validate_plan(broken)
        with self.assertRaisesRegex(ValueError, "rejected"):
            validate_plan({"status": "rejected"})
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "rgb.png"
            Image.new("RGB", (10, 10)).save(image)
            config = {"VLM_MODEL": "test", "VLM_BASE_URL": "https://example.invalid/v1", "VLM_API_KEY": "not-a-real-key"}
            response = MagicMock()
            response.__enter__.return_value = response
            with patch("agentic_manipulation.vlm.urllib.request.build_opener") as opener:
                opener.return_value.open.return_value = response
                response.read.return_value = json.dumps({"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(plan)}}]}).encode()
                self.assertEqual(backend_plan("搬运", image, config), plan)
                self.assertEqual(locate(image, task_identity(plan), config), plan)
                locate_prompt = json.loads(opener.return_value.open.call_args.args[0].data)["messages"][0]["content"]
                self.assertIn('"action": "pick_place"', locate_prompt)
                self.assertNotIn('pick|pick_place', locate_prompt)
                self.assertEqual(opener.return_value.open.call_count, 2)
                planned = plan_task("搬运", image, config)
                self.assertEqual(planned["steps"], ["LOCATE_OBJECT", "LOCATE_DESTINATION", "GRASP", "LIFT", "PLACE"])
                self.assertEqual(plan_task("搬运", image, config, identity=task_identity(plan))["object"], plan["object"])
                self.assertEqual(plan_task("搬运", image, config, held_object="red")["steps"], ["LOCATE_DESTINATION", "PLAN_PLACE", "PLACE"])
                with self.assertRaisesRegex(ValueError, "same cube"):
                    plan_task("搬运", image, config, held_object="blue")
                with self.assertRaisesRegex(ValueError, "identity"):
                    plan_task("搬运", image, config, identity=("pick_place", "blue", "yellow"))
                verdict = {"status": "failure", "reason": "still held", "recovery": "replace"}
                response.read.return_value = json.dumps({"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(verdict)}}]}).encode()
                self.assertEqual(verify("搬运", image, plan, config), verdict)
                self.assertEqual(validate_verdict(dict(verdict, status="uncertain", recovery=None))["status"], "uncertain")
                for broken in (None, {}, dict(verdict, status="maybe"), dict(verdict, reason=""),
                               dict(verdict, recovery="delete"), dict(verdict, status="success")):
                    with self.assertRaises(ValueError):
                        validate_verdict(broken)
                request = opener.return_value.open.call_args.args[0]
                self.assertIn("data:image/png;base64,", request.data.decode())
                self.assertEqual(json.loads(request.data)["response_format"], {"type": "json_object"})
                response.read.return_value = json.dumps({"choices": [{"finish_reason": "stop", "message": {"content": "```json\n" + json.dumps(verdict) + "\n```"}}]}).encode()
                with self.assertRaisesRegex(ValueError, "malformed"):
                    verify("搬运", image, plan, config)
                response.read.return_value = b'{"choices":[{"finish_reason":"length"}]}'
                with self.assertRaisesRegex(ValueError, "incomplete"):
                    backend_plan("搬运", image, config)
                for body in (b'not json', b'{"choices":[null]}'):
                    response.read.return_value = body
                    with self.assertRaisesRegex(ValueError, "malformed"):
                        backend_plan("搬运", image, config)
                for error in (urllib.error.URLError("secret"), http.client.IncompleteRead(b"partial")):
                    opener.return_value.open.side_effect = error
                    with self.assertRaisesRegex(ConnectionError, "connection failed"):
                        backend_plan("搬运", image, config)

    def test_visual_grounding_uses_box_and_color(self):
        rgb = np.full((40, 40, 3), [205, 128, 82], dtype=np.uint8)
        rgb[5:10, 5:10] = [0, 0, 255]
        rgb[25:30, 25:30] = [0, 0, 255]
        mask, box = grounded_region(rgb, {"color": "blue", "bbox": [0, 0, 500, 500]})
        self.assertEqual(box, [5, 5, 10, 10])
        self.assertEqual(int(mask.sum()), 25)
        clipped, _ = grounded_region(rgb, {"color": "blue", "bbox": [150, 150, 250, 250]})
        np.testing.assert_array_equal(clipped, mask)
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            grounded_region(rgb, {"color": "blue", "bbox": [0, 0, 1000, 1000]})
        with self.assertRaisesRegex(ValueError, "no usable"):
            grounded_region(rgb, {"color": "red", "bbox": [0, 0, 500, 500]})


if __name__ == "__main__":
    unittest.main()
