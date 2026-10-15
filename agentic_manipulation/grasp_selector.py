"""双环境抓取筛选：模型碰撞检查在 am-grasp，机器人约束在 am-sim。"""

import numpy as np
from scipy.spatial.transform import Rotation

from .geometry import panda_tcp_from_grasp
from .placement_planner import plan_place


class ExecutionFailure(RuntimeError):
    """可由新观测恢复的执行失败；碰撞与超时仍立即停止。"""


def filter_candidates(grasps, scene):
    from collision_detector import ModelFreeCollisionDetector

    counts = {"raw_candidates": len(grasps)}
    finite = np.isfinite(grasps.grasp_group_array).all(axis=1)
    grasps = grasps[finite & (grasps.scores > 0)]
    counts["score_feasible_candidates"] = len(grasps)
    grasps = grasps[(grasps.widths > 0) & (grasps.widths <= 0.08)]
    counts["width_feasible_candidates"] = len(grasps)
    if len(grasps):
        detector = ModelFreeCollisionDetector(scene, voxel_size=0.005)
        keep = []
        for first in range(0, len(grasps), 32):
            collision, empty = detector.detect(grasps[first:first + 32], approach_dist=0.05,
                                                collision_thresh=0.01, return_empty_grasp=True)
            keep.extend((~collision & ~empty).tolist())
        # IK 前保留相邻姿态，避免通用去重丢掉 Panda 可执行的候选。
        grasps = grasps[np.asarray(keep)]
        grasps.sort_by_score()
    counts["filtered_candidates"] = len(grasps)
    counts.update(score_rejected=counts["raw_candidates"] - counts["score_feasible_candidates"],
                  width_rejected=counts["score_feasible_candidates"] - counts["width_feasible_candidates"],
                  collision_rejected=counts["width_feasible_candidates"] - counts["filtered_candidates"])
    return grasps, counts


def select_grasp(candidates, target_points, current, reachable, destination=None, diagnostics=None, observe_held=False):
    if candidates.ndim != 2 or candidates.shape[1] != 17 or not np.isfinite(candidates).all():
        raise ValueError("invalid grasp candidates")
    if target_points.ndim != 2 or target_points.shape[1] != 3 or not len(target_points) or not np.isfinite(target_points).all():
        raise ValueError("invalid target point cloud")
    counts = diagnostics if diagnostics is not None else {}
    counts.update(candidates=len(candidates), opening=0, orientation=0, centering=0, table=0, ik=0)
    center = (target_points.min(axis=0) + target_points.max(axis=0)) / 2
    half_extent = (target_points.max(axis=0) - target_points.min(axis=0)) / 2
    for rank, index in enumerate(np.argsort(-candidates[:, 0], kind="stable"), 1):
        grasp = candidates[index]
        if grasp[0] <= 0 or not 0 < grasp[1] <= 0.08:
            counts["opening"] += 1
            continue
        target = panda_tcp_from_grasp(grasp)
        if target[2, 2] > -0.5 or target[2, 3] < 0.012:
            counts["orientation"] += 1
            continue
        if np.any(np.abs(target[:2, 3] - center[:2]) > half_extent[:2] * 0.75):
            counts["centering"] += 1
            continue
        flipped = target.copy()
        flipped[:3, :2] *= -1
        reason = "table"
        for target in sorted((target, flipped), key=lambda t: Rotation.from_matrix(current[:3, :3].T @ t[:3, :3]).magnitude()):
            low_pad = min((target @ [0, sign * 0.04758, 0.00025, 1])[2]
                          - np.abs(target[2, :3]) @ [0.00875, 0.0076, 0.00925] for sign in (-1, 1))
            if low_pad < 0.002:
                continue
            pregrasp = target.copy()
            pregrasp[:3, 3] -= target[:3, 2] * 0.03
            high = target.copy()
            high[2, 3] = 0.25
            lifted = target.copy()
            lifted[2, 3] += 0.12
            safe = current.copy()
            safe[2, 3] = max(safe[2, 3], 0.25)
            placement = None if destination is None else plan_place(target, center, destination)
            waypoints = [safe, high, pregrasp, target, lifted]
            if placement is not None:
                waypoints.extend(placement.values())
            observation = plan_place(target, center, center)["upright_pose"] if observe_held else None
            if observation is not None:
                waypoints.append(observation)
            if reachable(waypoints):
                counts["selected_index"] = int(index)
                counts["selected_rank"] = rank
                return {"index": int(index), "target": target, "pregrasp": pregrasp,
                        "high": high, "lift": lifted, "safe": safe, "center": center,
                        "placement": placement, "observation": observation}
            reason = "ik"
        counts[reason] += 1
    raise ExecutionFailure("no candidate passes Panda opening, table clearance and IK checks")
