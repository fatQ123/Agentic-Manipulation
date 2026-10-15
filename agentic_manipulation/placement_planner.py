"""从当前视觉观测生成世界坐标系放置关键点，不执行机器人动作。"""

from pathlib import Path

import numpy as np
from PIL import Image

from .geometry import backproject, grounded_region, placement_pose, transform_points


def locate_destination(capture, target):
    capture = Path(capture)
    with Image.open(capture / "rgb.png") as image:
        mask, _ = grounded_region(np.asarray(image.convert("RGB")), target)
    points, _ = backproject(np.load(capture / "depth.npy", allow_pickle=False),
                            np.load(capture / "camera_intrinsic.npy", allow_pickle=False), mask)
    points = transform_points(points, np.linalg.inv(np.load(capture / "camera_extrinsic.npy", allow_pickle=False)))
    destination = (points.min(axis=0) + points.max(axis=0)) / 2
    destination[2] = points[:, 2].max() + 0.035
    if not (-0.25 < destination[0] < 0.3 and abs(destination[1]) < 0.35 and 0.06 < destination[2] < 0.15):
        raise ValueError("destination outside tabletop workspace")
    return destination


def plan_place(grasp_pose, visual_center, destination):
    place = placement_pose(grasp_pose, visual_center, destination)
    transit = place.copy()
    transit[2, 3] = 0.25
    raised = grasp_pose.copy()
    raised[2, 3] = 0.25
    aligned = place.copy()
    aligned[:3, 3] = raised[:3, 3]
    return {"lift_pose": raised, "upright_pose": aligned, "pre_place_pose": transit,
            "place_pose": place, "release_pose": place.copy(), "retreat_pose": transit.copy()}
