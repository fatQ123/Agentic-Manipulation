"""单次视觉抓取：同一仿真中观察、GraspNet 子进程预测、执行并独立判分。"""

import argparse
import json
from pathlib import Path
import subprocess
import time

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from .capture import make_env, save_observation
from .grasp_selector import ExecutionFailure, select_grasp
from .placement_planner import locate_destination, plan_place
from .resource_manager import ResourceManager
from .task_planner import plan_task
from .pointcloud import reconstruct


class PickController:
    def __init__(self, env, writer, target_actor=None, deadline=None, notify=None):
        self.env = env
        self.notify = notify
        self.deadline = deadline
        self.base = env.unwrapped
        self.agent = self.base.agent
        self.writer = writer
        self.steps = 0
        self.robot_links = {link.name for link in self.agent.robot.get_links()}
        self.target_actor = self.base.cube if target_actor is None else target_actor
        self.target_name = self.target_actor._objs[0].name

    def pose(self):
        return self.agent.tcp.pose.to_transformation_matrix()[0].cpu().numpy()

    def action(self, world_pose, gripper):
        base_pose = self.agent.robot.pose.to_transformation_matrix()[0].cpu().numpy()
        target = np.linalg.inv(base_pose) @ world_pose
        return np.r_[target[:3, 3], Rotation.from_matrix(target[:3, :3]).as_euler("XYZ"), gripper].astype(np.float32)

    def step(self, action):
        if self.deadline is not None and time.monotonic() >= self.deadline:
            raise TimeoutError("task deadline exceeded")
        if self.notify is not None:
            self.notify("before_step", {})
        obs, _, _, truncated, info = self.env.step(action)
        self.steps += 1
        self.writer.append_data(obs["sensor_data"]["base_camera"]["rgb"][0].cpu().numpy())
        for contact in self.base.scene.get_contacts():
            names = {body.entity.name for body in contact.bodies}
            robot = names & self.robot_links
            if not robot or robot == {"panda_link0"}:
                continue
            impulse = sum(np.linalg.norm(point.impulse) for point in contact.points)
            allowed_grip = self.target_name in names and robot <= {"panda_leftfinger", "panda_rightfinger"}
            if impulse > 1e-3 and not allowed_grip:
                raise RuntimeError(f"unexpected robot contact: {sorted(names)}")
        if self.notify is not None:
            self.notify("frame", {"obs": obs})
        # PickCube 的随机目标球成功信号不属于本任务；仅采用本项目抬升判分。
        if bool(info.get("fail", False)) or bool(truncated):
            raise RuntimeError("simulation episode ended before pick evaluation")

    def move(self, target, gripper=1):
        start = self.pose()
        rotations = Rotation.from_matrix(np.stack([start[:3, :3], target[:3, :3]]))
        angle = (rotations[1] * rotations[0].inv()).magnitude()
        count = max(1, int(np.ceil(np.linalg.norm(target[:3, 3] - start[:3, 3]) / 0.005)), int(np.ceil(angle / 0.08)))
        interpolate = Slerp([0, 1], rotations)
        for fraction in np.linspace(0, 1, count + 1)[1:]:
            pose = np.eye(4)
            pose[:3, 3] = start[:3, 3] * (1 - fraction) + target[:3, 3] * fraction
            pose[:3, :3] = interpolate(fraction).as_matrix()
            self.step(self.action(pose, gripper))
        for _ in range(40):
            current = self.pose()
            position_error = np.linalg.norm(current[:3, 3] - target[:3, 3])
            rotation_error = Rotation.from_matrix(current[:3, :3] @ target[:3, :3].T).magnitude()
            if position_error < 0.004 and rotation_error < 0.05:
                return
            self.step(self.action(target, gripper))
        raise RuntimeError(f"waypoint timeout: position={position_error:.4f} m, rotation={rotation_error:.3f} rad")

    def reachable(self, waypoints):
        import sapien
        from mani_skill.utils.structs.pose import Pose

        arm = self.agent.controller.controllers["arm"]
        base_pose = self.agent.robot.pose.to_transformation_matrix()[0].cpu().numpy()
        qpos = self.agent.robot.get_qpos().clone()
        for pose in waypoints:
            if self.deadline is not None and time.monotonic() >= self.deadline:
                raise TimeoutError("task deadline exceeded")
            joints = arm.kinematics.compute_ik(Pose.create(sapien.Pose(np.linalg.inv(base_pose) @ pose)), qpos)
            if joints is None:
                return False
            qpos[:, arm.active_joint_indices] = joints
        return True


def infer_grasps(output, grasp_python, checkpoint, seed, plan, resources):
    reconstruct(output / "capture", output / "cloud", None if plan is None else plan["object"])
    resources.run_grasp([
        str(Path(grasp_python).resolve()), "-m", "agentic_manipulation.predict_grasps",
        str((output / "cloud" / "points.npz").resolve()),
        "--output", str((output / "grasps").resolve()),
        "--checkpoint", str(Path(checkpoint).resolve()), "--seed", str(seed),
    ], output / "inference.log", Path(__file__).resolve().parents[1])
    candidates = np.load(output / "grasps" / "grasps_world.npy", allow_pickle=False)
    if candidates.ndim != 2 or candidates.shape[1] != 17 or not len(candidates):
        raise ValueError("invalid or empty grasp candidates")
    return candidates


def execute_pick(control, output, candidates, destination, report, seed, observe_held=False):
    report["selection"] = {}
    if control.notify is not None:
        control.notify("checking_path", {})
    with np.load(output / "cloud" / "points.npz", allow_pickle=False) as cloud:
        selected = select_grasp(candidates, cloud["target_world"], control.pose(), control.reachable,
                                destination, report["selection"], observe_held=observe_held)
    target, pregrasp, high = selected["target"], selected["pregrasp"], selected["high"]
    report["candidate_index"] = selected["index"]
    report["tcp_target_world"] = target.tolist()
    if selected["observation"] is not None:
        report["observation_pose"] = selected["observation"].tolist()
    if selected["placement"] is not None:
        report["placement_waypoints"] = {key: value.tolist() for key, value in selected["placement"].items()}
    if control.notify is not None:
        control.notify("selected_grasp", {"directory": output, "target": target, "report": report})
    safe = selected["safe"]
    for name, pose in [("raise", safe), ("transit", high), ("pregrasp", pregrasp), ("approach", target)]:
        print(f"seed {seed}: {name}", flush=True)
        if control.notify is not None:
            control.notify(name, {})
        control.move(pose)
        report["phases"].append(name)
    if control.notify is not None:
        control.notify("close", {})
    for _ in range(control.base.control_freq):
        control.step(control.action(target, -1))
    if not bool(control.agent.is_grasping(control.target_actor)):
        raise ExecutionFailure("fingers did not establish a grasp")
    report["phases"].append("close")
    lift = selected["lift"]
    if control.notify is not None:
        control.notify("lift", {})
    control.move(lift, -1)
    report["phases"].append("lift")
    if control.notify is not None:
        control.notify("hold", {})
    min_bottom = float("inf")
    hold_success = True
    for _ in range(control.base.control_freq):
        control.step(control.action(lift, -1))
        # 真实物体状态只用于独立验收，不参与目标定位和动作生成。
        cube_pose = control.target_actor.pose.to_transformation_matrix()[0].cpu().numpy()
        bottom = float(cube_pose[2, 3] - control.base.cube_half_size * np.abs(cube_pose[2, :3]).sum())
        min_bottom = min(min_bottom, bottom)
        hold_success &= bottom >= 0.05 and bool(control.agent.is_grasping(control.target_actor))
    report["phases"].append("hold")
    report.update(min_bottom_during_hold_m=min_bottom, hold_s=1.0, pick_hold_success=bool(hold_success))
    return target, selected["center"]


def execute_place(control, target, visual_center, destination, bowl_color, report):
    if "placement_waypoints" in report:
        poses = {key: np.asarray(value) for key, value in report["placement_waypoints"].items()}
    else:
        poses = plan_place(target, visual_center, destination)
        report["placement_waypoints"] = {key: value.tolist() for key, value in poses.items()}
    if control.notify is not None:
        control.notify("checking_path", {})
    if not control.reachable(list(poses.values())):
        raise ExecutionFailure("placement path is not reachable")
    raised, aligned, transit, place = (poses[key] for key in ("lift_pose", "upright_pose", "pre_place_pose", "place_pose"))
    if control.notify is not None:
        control.notify("place", {})
    control.move(raised, -1)
    control.move(aligned, -1)
    control.move(transit, -1)
    control.move(place, -1)
    report["phases"].append("place")
    if control.notify is not None:
        control.notify("release_retreat", {})
    for _ in range(control.base.control_freq):
        control.step(control.action(poses["release_pose"], 1))
    control.move(poses["retreat_pose"], 1)
    report["phases"].append("release_retreat")
    bowl = control.base.bowls[bowl_color]
    place_success = True
    for _ in range(control.base.control_freq):
        control.step(control.action(transit, 1))
        # 真值只用于独立判分：方块整体在碗内、脱爪、低速度保持一秒。
        cube_pose = control.target_actor.pose.to_transformation_matrix()[0].cpu().numpy()
        bowl_center = bowl.pose.p[0].cpu().numpy()
        extent = control.base.cube_half_size * np.abs(cube_pose[:3, :3]).sum(axis=1)
        delta = np.abs(cube_pose[:2, 3] - bowl_center[:2]) + extent[:2]
        bottom = cube_pose[2, 3] - extent[2]
        speed = np.linalg.norm(control.target_actor.linear_velocity[0].cpu().numpy())
        spin = np.linalg.norm(control.target_actor.angular_velocity[0].cpu().numpy())
        place_success &= bool(np.all(delta <= 0.07) and 0.005 < bottom < 0.02 and speed <= 0.02
                              and spin <= 0.2 and not bool(control.agent.is_grasping(control.target_actor)))
    report["phases"].append("place_hold")
    report.update(place_hold_s=1.0, place_hold_success=bool(place_success),
                  final_cube_position=cube_pose[:3, 3].tolist(), bowl_position=bowl_center.tolist())


def run_pick(output, grasp_python, checkpoint, seed=0, instruction=None):
    import imageio.v2 as imageio

    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    report = {"seed": seed, "success": False, "phases": []}
    env = None
    writer = None
    start = time.monotonic()
    resources = ResourceManager()
    try:
        task = "PickCube-v1" if instruction is None else "LanguageTabletop-v2"
        env = make_env(task=task)
        obs, _ = env.reset(seed=seed)
        save_observation(obs, output / "capture", seed, task)
        plan = None
        destination = None
        target_actor = env.unwrapped.cube
        if instruction is not None:
            from .vlm import configuration
            config = configuration()
            report.update(instruction=instruction, model=config["VLM_MODEL"])
            plan = resources.call_vlm(plan_task, instruction, output / "capture" / "rgb.png", config, trial=report, stage="plan")
            report["plan"] = plan
            (output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
            target_actor = env.unwrapped.cubes[plan["object"]["color"]]
            if plan["action"] == "pick_place":
                destination = locate_destination(output / "capture", plan["destination"])
                report["destination_world"] = destination.tolist()
        candidates = infer_grasps(output, grasp_python, checkpoint, seed, plan, resources)
        writer = imageio.get_writer(output / "pick.mp4", fps=env.unwrapped.control_freq)
        control = PickController(env, writer, target_actor)
        with resources.phase("execute"):
            target, visual_center = execute_pick(control, output, candidates, destination, report, seed)
            if destination is not None:
                execute_place(control, target, visual_center, destination, plan["destination"]["color"], report)
        report.update(success=report["pick_hold_success"] and report.get("place_hold_success", True), steps=control.steps)
        if not report["success"]:
            report["failure"] = "independent lift/place evaluation failed"
    except (OSError, ValueError, RuntimeError, ImportError, subprocess.SubprocessError) as exc:
        report["failure"] = str(exc)
    finally:
        resources.close()
        report["resources"] = resources.report
        try:
            if writer is not None:
                writer.close()
        finally:
            if env is not None:
                env.close()
            report["elapsed_s"] = time.monotonic() - start
            (output / "result.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--grasp-python", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--instruction", help="V2 自然语言指令；省略时运行 V1")
    parser.add_argument("--v3", action="store_true", help="视觉反馈闭环，最多执行三轮")
    args = parser.parse_args()
    if args.v3 and not args.instruction:
        parser.error("--v3 requires --instruction")
    try:
        if args.v3:
            from .closed_loop import run_closed_loop
            report = run_closed_loop(args.output, args.grasp_python, args.checkpoint, args.seed, args.instruction)
        else:
            report = run_pick(args.output, args.grasp_python, args.checkpoint, args.seed, args.instruction)
    except OSError as exc:
        parser.exit(1, f"抓取启动失败：{exc}\n")
    print(json.dumps(report, indent=2))
    if not report["success"]:
        parser.exit(1)


if __name__ == "__main__":
    main()
