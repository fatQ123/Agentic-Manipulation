"""在 am-grasp 独立进程中运行官方 GraspNet，输出 Panda 可用的抓取候选。"""

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

from .geometry import transform_points
from .grasp_selector import filter_candidates


def predict(source, output, checkpoint, seed=0):
    import torch
    from graspnetAPI import GraspGroup

    root = Path(__file__).resolve().parents[1] / "third_party" / "graspnet-baseline"
    if not (root / "models" / "graspnet.py").is_file():
        raise FileNotFoundError("缺少 GraspNet 源码，请参照 docs/graspnet-setup.md 手动准备")
    sys.path.insert(0, str(root / "models"))
    sys.path.insert(0, str(root / "utils"))
    from graspnet import GraspNet, pred_decode

    if not torch.cuda.is_available():
        raise RuntimeError("GraspNet CUDA 扩展需要可用 GPU")
    with np.load(source, allow_pickle=False) as data:
        target = data["target_camera"]
        scene = data["scene_camera"]
        world_from_camera = data["world_from_camera"]
    for points in (target, scene):
        if points.ndim != 2 or points.shape[1] != 3 or not len(points) or not np.isfinite(points).all():
            raise ValueError("点云必须是非空、有限的 N×3 米制相机坐标")
    transform_points(target[:1], world_from_camera)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    torch.cuda.reset_peak_memory_stats()
    net = GraspNet(is_training=False).cuda().eval()
    # 仅加载用户从官方来源提供的 checkpoint；不下载权重。
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    net.load_state_dict(state["model_state_dict"])
    indices = rng.choice(len(target), 20000, replace=len(target) < 20000)
    samples = np.ascontiguousarray(target[indices][None], dtype=np.float32)
    with torch.no_grad():
        predictions = pred_decode(net({"point_clouds": torch.from_numpy(samples).cuda()}))[0]
    grasps = GraspGroup(predictions.cpu().numpy())
    grasps, counts = filter_candidates(grasps, scene)
    world = GraspGroup(grasps.grasp_group_array.copy()).transform(world_from_camera)
    np.save(output / "grasps_camera.npy", grasps.grasp_group_array, allow_pickle=False)
    np.save(output / "grasps_world.npy", world.grasp_group_array, allow_pickle=False)
    torch.cuda.synchronize()
    report = {
        "seed": seed,
        "source": str(Path(source).resolve()),
        "checkpoint": str(Path(checkpoint).resolve()),
        "length_unit": "m",
        **counts,
        "elapsed_s": time.monotonic() - start,
        "peak_torch_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "peak_torch_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
        "status": "candidates_ready" if len(grasps) else "no_valid_grasp",
        "limitation": "single-view generic gripper collision screen; robot IK and contacts still required",
    }
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="pointcloud 输出的 points.npz")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    try:
        report = predict(args.source, args.output, args.checkpoint, args.seed)
        print(json.dumps(report, indent=2))
        if report["status"] != "candidates_ready":
            parser.exit(1, "没有有效抓取候选，禁止继续执行动作。\n")
    except (OSError, ValueError, RuntimeError, ImportError, KeyError) as exc:
        parser.exit(1, f"抓取预测失败：{exc}\n")


if __name__ == "__main__":
    main()
