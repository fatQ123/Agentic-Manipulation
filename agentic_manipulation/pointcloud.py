"""从已保存的 RGB-D 重建红色目标与场景点云；不读取物体真实位姿。"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .geometry import backproject, red_region, transform_points, grounded_region


def reconstruct(source, output, target=None):
    source, output = Path(source), Path(output)
    metadata = json.loads((source / "metadata.json").read_text(encoding="utf-8"))
    if (
        metadata.get("depth_unit") != "m"
        or metadata.get("camera_frame") != "opencv"
        or metadata.get("extrinsic_direction") != "world_to_camera"
    ):
        raise ValueError("capture must specify meter depth, OpenCV camera and world_to_camera extrinsic")
    with Image.open(source / "rgb.png") as image:
        rgb = np.asarray(image.convert("RGB"))
    depth = np.load(source / "depth.npy", allow_pickle=False)
    intrinsic = np.load(source / "camera_intrinsic.npy", allow_pickle=False)
    extrinsic = np.load(source / "camera_extrinsic.npy", allow_pickle=False)
    if rgb.shape[:2] != depth.shape:
        raise ValueError("RGB and depth image sizes differ")
    mask, bbox = red_region(rgb) if target is None else grounded_region(rgb, target)
    scene_camera, scene_valid = backproject(depth, intrinsic)
    target_camera, target_valid = backproject(depth, intrinsic, mask)
    # extrinsic_cv 是世界→相机；反投影点进入世界坐标前必须求逆。
    transform_points(target_camera[:1], extrinsic)
    world_from_camera = np.linalg.inv(extrinsic)
    scene_world = transform_points(scene_camera, world_from_camera)
    target_world = transform_points(target_camera, world_from_camera)

    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(
        output / "points.npz",
        scene_camera=scene_camera,
        scene_world=scene_world,
        scene_rgb=rgb[scene_valid],
        target_camera=target_camera,
        target_world=target_world,
        target_rgb=rgb[target_valid],
        target_mask=target_valid,
        world_from_camera=world_from_camera,
    )
    overlay = Image.fromarray(rgb)
    x1, y1, x2, y2 = bbox
    ImageDraw.Draw(overlay).rectangle([x1, y1, x2 - 1, y2 - 1], outline=(0, 255, 0), width=2)
    overlay.save(output / "target_bbox.png")
    Image.fromarray(target_valid.astype(np.uint8) * 255).save(output / "target_mask.png")
    summary = {
        "source": str(source.resolve()),
        "target": "red_cube" if target is None else target["color"] + "_cube",
        "bbox_xyxy_exclusive": bbox,
        "length_unit": "m",
        "camera_frame": "opencv",
        "scene_points": len(scene_camera),
        "target_points": len(target_camera),
        "target_center_world": target_world.mean(axis=0).tolist(),
        "target_bounds_world": [target_world.min(axis=0).tolist(), target_world.max(axis=0).tolist()],
        "limitation": "visible surface centroid, not object pose or grasp pose",
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return summary


def view_cloud(path):
    try:
        import open3d as o3d
    except ImportError as exc:
        raise RuntimeError("查看窗口需要 Open3D，请按 docs/setup.md 手动安装；点云文件已保存。") from exc
    with np.load(path, allow_pickle=False) as data:
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(data["target_world"])
        cloud.colors = o3d.utility.Vector3dVector(data["target_rgb"] / 255.0)
    o3d.visualization.draw_geometries([cloud], window_name="Target point cloud (world, meters)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="capture 输出目录")
    parser.add_argument("--output", type=Path, required=True, help="新的点云输出目录")
    parser.add_argument("--view", action="store_true", help="用 Open3D 查看目标点云")
    args = parser.parse_args()
    try:
        summary = reconstruct(args.source, args.output)
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        if args.view:
            view_cloud(args.output / "points.npz")
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        parser.exit(1, f"点云处理失败：{exc}\n")


if __name__ == "__main__":
    main()
