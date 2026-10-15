"""相机几何：长度统一为米，T_A_B 将 B 坐标转换为 A 坐标。"""

import numpy as np
from scipy.ndimage import label


def depth_to_meters(depth_mm):
    """ManiSkill minimal shader 的毫米深度；无效深度统一为 0。"""
    depth = np.asarray(depth_mm, dtype=np.float32)
    if depth.ndim != 2:
        raise ValueError("depth must have shape (H, W)")
    return np.where(np.isfinite(depth) & (depth > 0), depth / 1000.0, 0)


def red_region(rgb):
    """返回最大红色连通区域及右／下边界不包含在内的目标框。"""
    rgb = np.asarray(rgb)
    if rgb.ndim != 3 or rgb.shape[-1] != 3 or rgb.dtype != np.uint8:
        raise ValueError("rgb must be uint8 with shape (H, W, 3)")
    red, green, blue = rgb.astype(np.float32).transpose(2, 0, 1)
    # shortcut: 最大红色连通区域视作唯一目标，多目标语义定位在 V2 接入 VLM。
    regions, count = label((red >= 80) & (red > 2 * green) & (red > 2 * blue))
    if count == 0:
        raise ValueError("no red target found in RGB image")
    sizes = np.bincount(regions.ravel())
    sizes[0] = 0
    mask = regions == sizes.argmax()
    v, u = np.nonzero(mask)
    bbox = [int(u.min()), int(v.min()), int(u.max()) + 1, int(v.max()) + 1]
    return mask, bbox


def backproject(depth_m, intrinsic, mask=None):
    """返回 OpenCV 相机坐标系中的 N×3 点，以及对应的有效像素掩码。"""
    depth = np.asarray(depth_m)
    intrinsic = np.asarray(intrinsic, dtype=np.float64)
    if depth.ndim != 2:
        raise ValueError("depth must have shape (H, W)")
    if (
        intrinsic.shape != (3, 3)
        or not np.isfinite(intrinsic).all()
        or intrinsic[0, 0] <= 0
        or intrinsic[1, 1] <= 0
        or not np.allclose(intrinsic[2], [0, 0, 1])
        or not np.isclose(intrinsic[1, 0], 0)
    ):
        raise ValueError("invalid pinhole camera intrinsic matrix")
    valid = np.isfinite(depth) & (depth > 0)
    if mask is not None:
        mask = np.asarray(mask)
        if mask.shape != depth.shape or mask.dtype != np.bool_:
            raise ValueError("mask must be boolean and match depth shape")
        valid &= mask
    v, u = np.nonzero(valid)
    if len(u) == 0:
        raise ValueError("no valid depth points in selected region")
    pixels = np.stack([u, v, np.ones_like(u)])
    rays = np.linalg.solve(intrinsic, pixels).T
    return (rays * depth[valid, None]).astype(np.float32), valid


def transform_points(points, transform):
    """用刚体变换 T_to_from 转换 N×3 点，不改变长度单位。"""
    points = np.asarray(points)
    transform = np.asarray(transform, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("points must be finite with shape (N, 3)")
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError("transform must be finite with shape (4, 4)")
    rotation = transform[:3, :3]
    if (
        not np.allclose(transform[3], [0, 0, 0, 1], atol=1e-6)
        or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5)
        or not np.isclose(np.linalg.det(rotation), 1, atol=1e-5)
    ):
        raise ValueError("transform must be a rigid SE(3) transform")
    return (points @ rotation.T + transform[:3, 3]).astype(np.float32)


def panda_tcp_from_grasp(grasp):
    """将一个世界坐标 GraspNet 17 维候选转为 Panda TCP 位姿。"""
    grasp = np.asarray(grasp)
    if grasp.shape != (17,) or not np.isfinite(grasp).all() or not 0 < grasp[1] <= 0.08:
        raise ValueError("invalid grasp or width beyond Panda's 80 mm opening")
    if not 0 < grasp[3] <= 0.04:
        raise ValueError("invalid GraspNet insertion depth")
    rotation = grasp[4:13].reshape(3, 3)
    pose = np.eye(4)
    # GraspNet x=接近、y=闭合；Panda TCP z=接近、y=闭合。
    pose[:3, :3] = np.column_stack((-rotation[:, 2], rotation[:, 1], rotation[:, 0]))
    # URDF 指尖最前端在 TCP 前方 9.5 mm，预测 depth 指向指尖而非 TCP。
    pose[:3, 3] = grasp[13:16] + rotation[:, 0] * (grasp[3] - 0.0095)
    transform_points(np.zeros((1, 3)), pose)
    return pose


def grounded_region(rgb, target):
    """用 VLM 框约束颜色连通域，避免将框内桌面送入 GraspNet。"""
    from .vlm import COLORS, BOWLS, validate_target

    validate_target(target, COLORS | BOWLS)
    color = target["color"]
    h, w = rgb.shape[:2]
    x1, y1, x2, y2 = np.array(target["bbox"]) * [w, h, w, h] / 1000
    x1, y1, x2, y2 = int(np.floor(x1)), int(np.floor(y1)), int(np.ceil(x2)), int(np.ceil(y2))
    r, g, b = rgb.astype(float).transpose(2, 0, 1)
    colors = {
        "red": (r >= 80) & (r > 2 * g) & (r > 2 * b),
        "green": (g >= 60) & (g > 1.6 * r) & (g > 1.6 * b),
        "blue": (b >= 80) & (b > 1.6 * r) & (b > 1.6 * g),
        "yellow": (r >= 80) & (g >= 60) & (r > 2 * b) & (g > 2 * b),
        "purple": (r >= 60) & (b >= 80) & (r > 1.6 * g) & (b > 1.6 * g),
    }
    # 以框内重叠选择整个连通域，避免模型框轻微截边损坏目标几何。
    regions, count = label(colors[color])
    sizes = np.bincount(regions[y1:y2, x1:x2].ravel(), minlength=count + 1)
    sizes[0] = 0
    if not count or sizes.max() < 12:
        raise ValueError("VLM bbox contains no usable matching object")
    if np.count_nonzero(sizes >= max(12, sizes.max() * 0.25)) > 1:
        raise ValueError("VLM bbox has multiple matching regions; ambiguous target")
    mask = regions == sizes.argmax()
    v, u = np.nonzero(mask)
    return mask, [int(u.min()), int(v.min()), int(u.max()) + 1, int(v.max()) + 1]


def placement_pose(grasp_pose, visual_center, destination):
    """搬运时转为竖直放置，按夹持坐标系中的观测偏移保持物体中心对齐。"""
    pose = np.eye(4)
    closing = grasp_pose[:3, 1].copy()
    closing[2] = 0
    if np.linalg.norm(closing) < 1e-6:
        raise ValueError("cannot orient vertical placement from this grasp")
    closing /= np.linalg.norm(closing)
    pose[:3, :3] = np.column_stack((np.cross(closing, [0, 0, -1]), closing, [0, 0, -1]))
    offset = grasp_pose[:3, :3].T @ (visual_center - grasp_pose[:3, 3])
    pose[:3, 3] = destination - pose[:3, :3] @ offset
    return pose
