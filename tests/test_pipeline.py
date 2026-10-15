"""最少必要用例：相机几何、目标提取、文件链路和夹爪标定。"""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from agentic_manipulation.geometry import backproject, depth_to_meters, red_region, transform_points, panda_tcp_from_grasp, placement_pose
from agentic_manipulation.pointcloud import reconstruct
from agentic_manipulation.placement_planner import plan_place


class PipelineTests(unittest.TestCase):
    def test_panda_axes_and_fingertip_depth(self):
        rotation = np.array([[0, 1, 0], [0, 0, -1], [-1, 0, 0]])
        grasp = np.r_[0.9, 0.06, 0.02, 0.03, rotation.ravel(), [0.1, 0.2, 0.04], -1]
        tcp = panda_tcp_from_grasp(grasp)
        np.testing.assert_allclose(tcp[:3, 2], [0, 0, -1])
        np.testing.assert_allclose(tcp[:3, 1], [1, 0, 0])
        np.testing.assert_allclose(tcp[:3, 3], [0.1, 0.2, 0.0195])
        np.testing.assert_allclose((tcp @ [0, 0, 0.0095, 1])[:3], [0.1, 0.2, 0.01])
        tilted = tcp.copy()
        angle = 0.4
        tilt = np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]])
        tilted[:3, :3] = tilt @ tcp[:3, :3]
        center, destination = np.array([0.1, 0.2, 0.02]), np.array([0.05, -0.1, 0.09])
        place = placement_pose(tilted, center, destination)
        offset = tilted[:3, :3].T @ (center - tilted[:3, 3])
        np.testing.assert_allclose(place[:3, :3] @ offset + place[:3, 3], destination)
        np.testing.assert_allclose(place[:3, 2], [0, 0, -1])
        transform_points(np.zeros((1, 3)), place)
        poses = plan_place(tilted, center, destination)
        self.assertEqual(list(poses), ["lift_pose", "upright_pose", "pre_place_pose", "place_pose", "release_pose", "retreat_pose"])
        np.testing.assert_allclose(poses["place_pose"], place)
        self.assertEqual(poses["pre_place_pose"][2, 3], 0.25)
        np.testing.assert_allclose(poses["retreat_pose"], poses["pre_place_pose"])
        grasp[1] = 0.09
        with self.assertRaisesRegex(ValueError, "opening"):
            panda_tcp_from_grasp(grasp)

    def test_metric_geometry_and_camera_world_direction(self):
        depth = depth_to_meters(np.array([[1000, 2000], [0, np.nan]]))
        intrinsic = np.array([[2, 0, 0], [0, 2, 0], [0, 0, 1]])
        points, valid = backproject(depth, intrinsic)
        np.testing.assert_allclose(points, [[0, 0, 1], [1, 0, 2]])
        np.testing.assert_array_equal(valid, [[True, True], [False, False]])
        camera_from_world = np.array([
            [0, -1, 0, 1], [1, 0, 0, 2], [0, 0, 1, 3], [0, 0, 0, 1]
        ])
        world = transform_points(points, np.linalg.inv(camera_from_world))
        np.testing.assert_allclose(world, [[-2, 1, -2], [-2, 0, -1]])
        np.testing.assert_allclose(transform_points(world, camera_from_world), points)

    def test_red_target_excludes_background_and_invalid_depth(self):
        rgb = np.array([
            [[255, 255, 255], [255, 0, 0], [200, 20, 10], [0, 0, 0], [255, 0, 0]],
            [[0, 0, 255], [255, 0, 0], [0, 255, 0], [205, 128, 82], [0, 0, 0]],
        ], dtype=np.uint8)
        mask, bbox = red_region(rgb)
        self.assertEqual(bbox, [1, 0, 3, 2])
        depth = np.ones((2, 5), dtype=np.float32)
        depth[0, 2] = 0
        points, valid = backproject(depth, np.eye(3), mask)
        np.testing.assert_allclose(points, [[1, 0, 1], [1, 1, 1]])
        self.assertEqual(int(valid.sum()), 2)

    def test_unusable_observations_stop_with_errors(self):
        with self.assertRaisesRegex(ValueError, "no red target"):
            red_region(np.zeros((2, 2, 3), dtype=np.uint8))
        with self.assertRaisesRegex(ValueError, "no valid depth"):
            backproject(np.array([[0, -1, np.nan, np.inf]]), np.eye(3))
        with self.assertRaisesRegex(ValueError, "intrinsic"):
            backproject(np.ones((2, 2)), np.zeros((3, 3)))
        with self.assertRaisesRegex(ValueError, "mask"):
            backproject(np.ones((2, 2)), np.eye(3), np.ones((1, 1), dtype=bool))
        with self.assertRaisesRegex(ValueError, "rigid"):
            transform_points(np.ones((1, 3)), np.diag([2, 1, 1, 1]))

    def test_saved_capture_to_cloud_without_overwriting_results(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "capture"
            output = Path(temporary) / "cloud"
            source.mkdir()
            rgb = np.zeros((3, 4, 3), dtype=np.uint8)
            rgb[1:3, 1:3] = [255, 0, 0]
            Image.fromarray(rgb).save(source / "rgb.png")
            np.save(source / "depth.npy", np.ones((3, 4), dtype=np.float32))
            np.save(source / "camera_intrinsic.npy", np.eye(3))
            extrinsic = np.eye(4)
            extrinsic[0, 3] = -10
            np.save(source / "camera_extrinsic.npy", extrinsic)
            metadata = {"depth_unit": "m", "camera_frame": "opencv", "extrinsic_direction": "world_to_camera"}
            (source / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

            summary = reconstruct(source, output)
            self.assertEqual(summary["target_points"], 4)
            self.assertEqual(summary["scene_points"], 12)
            with np.load(output / "points.npz", allow_pickle=False) as data:
                np.testing.assert_allclose(data["target_world"], [
                    [11, 1, 1], [12, 1, 1], [11, 2, 1], [12, 2, 1]
                ])
                np.testing.assert_array_equal(data["target_rgb"], np.tile([255, 0, 0], (4, 1)))
                np.testing.assert_allclose(data["world_from_camera"], np.linalg.inv(extrinsic))
            self.assertTrue((output / "target_bbox.png").is_file())
            self.assertTrue((output / "target_mask.png").is_file())
            summary_path = output / "summary.json"
            previous = summary_path.read_bytes()
            with self.assertRaises(FileExistsError):
                reconstruct(source, output)
            self.assertEqual(summary_path.read_bytes(), previous)
            metadata["depth_unit"] = "mm"
            (source / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "meter depth"):
                reconstruct(source, Path(temporary) / "wrong-unit")


if __name__ == "__main__":
    unittest.main()
