"""从 ManiSkill PickCube 的固定相机保存一帧 RGB-D，不安装或下载依赖。"""

import argparse
import importlib.util
import json
from importlib.metadata import version
from pathlib import Path

import numpy as np
from PIL import Image

from .geometry import backproject, depth_to_meters, transform_points


def make_env(human=False, task="PickCube-v1", max_episode_steps=None):
    missing = [
        name for name in ("torch", "gymnasium", "sapien", "mani_skill")
        if importlib.util.find_spec(name) is None
    ]
    if missing:
        raise RuntimeError(
            f"缺少依赖：{', '.join(missing)}。请按 docs/setup.md 手动准备环境；本程序不会安装依赖。"
        )

    import gymnasium as gym
    import mani_skill.envs  # 注册 PickCube-v1。

    if task == "LanguageTabletop-v2":
        from . import scene  # 注册 V2 场景。

    return gym.make(
        task,
        robot_uids="panda",
        num_envs=1,
        obs_mode="rgbd",
        control_mode="pd_ee_pose",
        max_episode_steps=max_episode_steps if max_episode_steps is not None else (1000 if task == "LanguageTabletop-v2" else 600),
        sim_backend="physx_cpu",
        render_backend="gpu",
        render_mode="human" if human else "rgb_array",
        sensor_configs={"base_camera": {"width": 640, "height": 480, "shader_pack": "minimal"}},
    )


def save_observation(obs, output, seed, task="PickCube-v1"):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    # 固定 num_envs=1；只去掉明确的批次与深度通道维度。
    sensor = obs["sensor_data"]["base_camera"]
    params = obs["sensor_param"]["base_camera"]
    rgb = sensor["rgb"][0].cpu().numpy()
    depth = depth_to_meters(sensor["depth"][0, ..., 0].cpu().numpy())
    intrinsic = params["intrinsic_cv"][0].cpu().numpy()
    # SAPIEN 返回 3×4 的 [R|t]，保存时补为齐次变换矩阵。
    extrinsic = np.eye(4, dtype=np.float32)
    extrinsic[:3] = params["extrinsic_cv"][0].cpu().numpy()
    if rgb.dtype != np.uint8 or rgb.shape != (*depth.shape, 3):
        raise ValueError("unexpected RGB-D shape or dtype; expected matching uint8 RGB and depth")
    points, valid = backproject(depth, intrinsic)
    transform_points(points[:1], extrinsic)

    Image.fromarray(rgb).save(output / "rgb.png")
    np.save(output / "depth.npy", depth, allow_pickle=False)
    np.save(output / "camera_intrinsic.npy", intrinsic, allow_pickle=False)
    np.save(output / "camera_extrinsic.npy", extrinsic, allow_pickle=False)
    metadata = {
        "environment": task,
        "robot": "panda",
        "camera": "base_camera",
        "seed": seed,
        "image_size": [int(rgb.shape[1]), int(rgb.shape[0])],
        "depth_unit": "m",
        "camera_frame": "opencv",
        "extrinsic_direction": "world_to_camera",
        "valid_depth_pixels": int(valid.sum()),
        "versions": {name: version(name) for name in ("mani_skill", "sapien", "torch", "numpy")},
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"已保存 RGB-D：{output.resolve()}，有效深度像素 {valid.sum()}")
    return metadata


def capture(output, seed=0, human=False):
    if Path(output).exists():
        raise FileExistsError(f"输出目录已存在：{output}")
    env = make_env(human)
    try:
        obs, _ = env.reset(seed=seed)
        metadata = save_observation(obs, output, seed)
        if human:
            viewer = env.render()
            print("关闭仿真窗口以结束；此阶段仅观察，不执行抓取。")
            while not viewer.closed:
                env.render()
    finally:
        env.close()
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="新的采集输出目录")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--human", action="store_true", help="打开仿真观察窗口")
    args = parser.parse_args()
    if args.seed < 0:
        parser.error("--seed must be non-negative")
    try:
        capture(args.output, args.seed, args.human)
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        parser.exit(1, f"采集失败：{exc}\n")


if __name__ == "__main__":
    main()
