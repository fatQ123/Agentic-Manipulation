"""持续桌面会话：终端逐条指令、SAPIEN 执行窗口和视觉调试窗口。"""

import argparse
from datetime import datetime
import json
from pathlib import Path
import select
import sys
import textwrap
import time

import numpy as np
from PIL import Image

from .capture import make_env
from .closed_loop import run_closed_loop
from .debug_view import DebugWindow, selected_grasp_data
from .progress import TerminalProgress


class DemoSession:
    def __init__(self, env, output, grasp_python, checkpoint, seed=0, notify=None):
        self.env = env.unwrapped
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=False)
        self.grasp_python, self.checkpoint, self.seed = grasp_python, checkpoint, seed
        self.notify = notify
        self.tasks = []
        self.events = []
        self.blocked = False
        self.reset()

    def save(self):
        (self.output / "session.json").write_text(json.dumps({"seed": self.seed, "tasks": self.tasks,
                                                            "events": self.events, "blocked": self.blocked,
                                                            "state": self.state, "held_object": self.held_object,
                                                            "tasks_since_reset": self.tasks_since_reset},
                                                           ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def reset(self):
        self.env.reset(seed=self.seed)
        self.observation_pose = self.env.agent.tcp.pose.to_transformation_matrix()[0].cpu().numpy()
        self.blocked = False
        self.state, self.held_object, self.tasks_since_reset = "IDLE", None, 0
        self.events.append({"command": "reset", "after_tasks": len(self.tasks)})
        self.save()
        if self.notify is not None:
            self.notify("reset", {"obs": self.env.get_obs()})

    def submit(self, instruction):
        if self.blocked:
            return {"success": False, "agent_status": "stopped", "state": "STOPPED", "failure": "会话已停止，请输入 reset 或 quit"}
        directory = self.output / f"task-{len(self.tasks) + 1:03d}"
        if self.notify is not None:
            self.notify("instruction", {"instruction": instruction})
        try:
            report = run_closed_loop(directory, self.grasp_python, self.checkpoint, self.seed, instruction,
                                     env=self.env, notify=self.notify, observation_pose=self.observation_pose)
        except KeyboardInterrupt:
            self.blocked = True
            self.state = "STOPPED"
            self.events.append({"command": "interrupted", "output": directory.name})
            self.save()
            raise
        self.blocked = report["state"] == "STOPPED"
        self.held_object = report["held_object"]
        self.state = "STOPPED" if self.blocked else "HOLDING" if self.held_object else "IDLE"
        self.tasks_since_reset += 1
        self.tasks.append({"instruction": instruction, "output": directory.name, "success": report["success"],
                           "agent_status": report["agent_status"], "steps": report.get("steps", 0),
                           "elapsed_s": report["elapsed_s"], "state": report["state"], "held_object": self.held_object})
        self.save()
        if self.notify is not None:
            self.notify("task_complete", {"directory": directory, "report": report, "blocked": self.blocked})
        return report

    def close(self):
        self.env.close()


class DemoView:
    def __init__(self, env, output):
        import imageio.v2 as imageio
        import matplotlib.pyplot as plt

        self.env, self.plt = env, plt
        plt.rcParams.update({"font.family": ["Noto Serif CJK SC", "DejaVu Sans"], "axes.unicode_minus": False})
        self.viewer = env.render()
        self.viewer.resolution = (960, 720)
        # 执行窗口只显示场景，避免原生编辑器面板遮住机器人。
        for plugin in self.viewer.plugins:
            plugin.close()
        self.viewer.plugins = []
        self.fig = plt.figure(figsize=(10, 7), layout="constrained")
        self.fig.canvas.manager.set_window_title("Agentic Manipulation · 视觉调试")
        self.rgb_ax = self.fig.add_subplot(221)
        self.depth_ax = self.fig.add_subplot(222)
        self.cloud_ax = self.fig.add_subplot(223, projection="3d")
        self.status_ax = self.fig.add_subplot(224)
        self.rgb_ax.set_title("当前 RGB")
        self.depth_ax.set_title("深度 / 米")
        self.rgb_artist = self.rgb_ax.imshow(np.zeros((480, 640, 3), dtype=np.uint8))
        self.depth_artist = self.depth_ax.imshow(np.zeros((480, 640)), vmin=0.2, vmax=1.4, cmap="viridis")
        self.rgb_ax.axis("off")
        self.depth_ax.axis("off")
        self.status_ax.axis("off")
        self.status_artist = self.status_ax.text(0, 1, "", va="top", fontsize=10)
        self.boxes = []
        self.instruction = "等待输入"
        self.recovery_reason = ""
        self.state, self.resource = "IDLE", "空闲"
        self.last_draw = 0
        self.writer = imageio.get_writer(Path(output) / "session.mp4", fps=env.control_freq)
        plt.show(block=False)
        self("reset", {"obs": env.get_obs()})

    def check(self):
        if self.viewer.closed or not self.plt.fignum_exists(self.fig.number):
            raise KeyboardInterrupt("demo window closed")

    def pump(self):
        self.check()
        self.env.render()
        self.fig.canvas.flush_events()
        self.check()

    def clear_boxes(self):
        for box in self.boxes:
            box.remove()
        self.boxes.clear()

    def frame(self, rgb, depth):
        self.rgb_artist.set_data(rgb)
        self.depth_artist.set_data(depth)
        self.writer.append_data(rgb)

    def __call__(self, stage, data):
        from matplotlib.patches import Rectangle

        self.check()
        if stage == "before_step":
            self.clear_boxes()
            self.pump()
            return
        if stage in ("frame", "reset"):
            sensor = data["obs"]["sensor_data"]["base_camera"]
            self.frame(sensor["rgb"][0].cpu().numpy(), sensor["depth"][0, ..., 0].cpu().numpy() / 1000)
        if stage == "frame":
            if time.monotonic() - self.last_draw < 0.15:
                return
        elif stage == "instruction":
            self.instruction = data["instruction"]
            self.recovery_reason = ""
            self.clear_boxes()
        elif stage == "reset":
            self.clear_boxes()
            self.cloud_ax.clear()
            self.cloud_ax.set_title("等待目标点云")
            self.instruction = "等待输入；reset 重置，quit 退出"
            self.recovery_reason = ""
            self.state, self.resource = "IDLE", "空闲"
        elif stage == "observation":
            self.clear_boxes()
            capture = data["directory"] / ("verification" if data.get("verification") else "capture")
            with Image.open(capture / "rgb.png") as image:
                self.frame(np.asarray(image.convert("RGB")), np.load(capture / "depth.npy", allow_pickle=False))
        elif stage == "plan_ready":
            self.clear_boxes()
            for key, color in (("object", "lime"), ("destination", "yellow")):
                target = data["plan"].get(key)
                if target:
                    x1, y1, x2, y2 = np.array(target["bbox"]) * [0.64, 0.48, 0.64, 0.48]
                    box = Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=color, linewidth=2)
                    self.rgb_ax.add_patch(box)
                    self.boxes.append(box)
        elif stage == "cloud_ready":
            self.cloud_ax.clear()
            with np.load(data["directory"] / "cloud" / "points.npz", allow_pickle=False) as cloud:
                all_points = cloud["scene_world"]
                visible = np.all((all_points >= [-.3, -.4, -.02]) & (all_points <= [.35, .4, .4]), axis=1)
                stride = max(1, int(visible.sum()) // 3000)
                scene = all_points[visible][::stride]
                self.cloud_ax.scatter(*scene.T, c=cloud["scene_rgb"][visible][::stride] / 255, s=2, alpha=0.6)
                target = cloud["target_world"]
                self.cloud_ax.scatter(*target.T, c=cloud["target_rgb"] / 255, s=4)
            self.cloud_ax.set(xlim=(-.3, .35), ylim=(-.4, .4), zlim=(-.02, .4), xlabel="X / m", ylabel="Y / m", zlabel="Z / m")
            self.cloud_ax.set_title("规划观测点云 / 世界坐标")
        elif stage == "selected_grasp":
            pose = data["target"]
            for axis, color in enumerate(("r", "g", "b")):
                self.cloud_ax.quiver(*pose[:3, 3], *pose[:3, axis], length=.07, color=color)
        elif stage == "state":
            self.state = data["state"]
            if self.state == "RECOVERING" and data.get("reason"):
                self.recovery_reason = "恢复原因：" + data["reason"]
        resource_labels = {"plan": "远程 VLM 请求中", "locate": "远程 VLM 重新定位中",
                           "verification": "远程 VLM 验证中", "grasp_inference": "GraspNet 子进程运行中",
                           "cloud_ready": "当前点云可用", "checking_path": "本地 IK 检查中",
                           "execute": "机器人控制中", "plan_ready": "VLM 请求完成", "verdict": "VLM 请求完成",
                           "task_complete": "外部调用已清理"}
        if stage in resource_labels:
            self.resource = resource_labels[stage]
        if stage in ("raise", "transit", "pregrasp", "approach", "close", "lift", "hold", "place", "release_retreat"):
            self.resource = "机器人控制中"
        if stage != "frame":
            labels = {"plan": "等待模型规划", "locate": "等待模型重新定位", "checking_path": "正在检查机器人路径",
                      "state": "任务状态更新", "grasp_inference": "等待 GraspNet 推理", "execute": "执行动作",
                      "verification": "等待视觉验证", "observe": "重新观察", "plan_ready": "目标已定位",
                      "cloud_ready": "点云已生成", "selected_grasp": "可执行抓取已选择", "place": "放置",
                      "raise": "升高", "transit": "转移", "pregrasp": "预抓取", "approach": "接近",
                      "close": "闭爪", "lift": "抬升", "hold": "稳定保持", "release_retreat": "释放与撤离",
                      "observation": "当前观测", "verdict": "视觉判断", "task_complete": "本条任务结束",
                      "reset": "场景已重置", "instruction": "任务已接收"}
            detail = self.recovery_reason
            if stage == "verdict":
                detail = data["verdict"]["status"] + "\n" + data["verdict"]["reason"]
                self.recovery_reason = ("恢复原因：" + data["verdict"]["reason"]
                                        if data["verdict"].get("recovery") in ("regrasp", "replace") else "")
            elif stage == "task_complete":
                report = data["report"]
                self.state = report["state"]
                detail = report.get("failure", report.get("stop_reason", ""))
                detail += "\n" + ("已停止，请 reset 或 quit" if data["blocked"] else "可以输入下一条指令")
            self.status_artist.set_text(textwrap.fill(self.instruction, 30) + "\n\n状态：" + self.state
                                        + "\n资源：" + self.resource + "\n" + labels.get(stage, stage)
                                        + "\n\n" + textwrap.fill(detail, 42))
        self.fig.canvas.draw()
        self.last_draw = time.monotonic()
        self.pump()
        if stage == "selected_grasp":
            self.fig.savefig(data["directory"] / "selected-grasp.png")
        elif stage == "task_complete":
            self.fig.savefig(data["directory"] / "debug.png")

    def close(self):
        try:
            self.writer.close()
        finally:
            self.plt.close(self.fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/demo") / datetime.now().strftime("session-%Y%m%d-%H%M%S-%f"))
    parser.add_argument("--grasp-python", type=Path, default=Path(sys.executable).resolve().parents[2] / "am-grasp/bin/python")
    parser.add_argument("--checkpoint", type=Path, default=Path("weights/checkpoint-rs.tar"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output directory already exists")
    env = session = view = debug = None
    try:
        print("Initializing simulation...", flush=True)
        env = make_env(human=True, task="LanguageTabletop-v2")
        session = DemoSession(env, args.output, args.grasp_python, args.checkpoint, args.seed)
        view = DemoView(session.env, args.output)
        debug = DebugWindow()
        progress = TerminalProgress()

        def notify(stage, data):
            if stage == "selected_grasp":
                data = selected_grasp_data(data)
            progress(stage, data)
            view(stage, data)
            debug(stage, data)

        session.notify = notify
        debug("reset", {"obs": session.env.get_obs()})
        print("Ready. 逐条输入任务；reset 重置，quit 退出。", flush=True)
        print("Task > ", end="", flush=True)
        while True:
            view.pump()
            if not select.select([sys.stdin], [], [], .05)[0]:
                continue
            line = sys.stdin.readline()
            if not line or line.strip().lower() in ("quit", "exit"):
                break
            instruction = line.strip()
            if instruction.lower() == "reset":
                session.reset()
            elif instruction:
                result = session.submit(instruction)
                print(json.dumps({key: result.get(key) for key in ("state", "success", "agent_status", "failure", "stop_reason")}, ensure_ascii=False), flush=True)
            print("Task > ", end="", flush=True)
    except KeyboardInterrupt:
        print("演示已停止。", flush=True)
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        parser.exit(1, f"演示失败：{exc}\n")
    finally:
        try:
            try:
                if debug is not None:
                    debug.close()
            finally:
                if view is not None:
                    view.close()
        finally:
            if session is not None:
                session.close()
            elif env is not None:
                env.close()


if __name__ == "__main__":
    main()
