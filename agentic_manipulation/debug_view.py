"""额外的三栏窗口：视觉定位、深度、投影到观测图上的 GraspNet 抓取。"""

import numpy as np
from PIL import Image


def selected_grasp_data(data):
    directory = data["directory"]
    candidates = np.load(directory / "grasps" / "grasps_world.npy", allow_pickle=False)
    grasp = candidates[data["report"]["candidate_index"]]
    extrinsic = np.load(directory / "capture" / "camera_extrinsic.npy", allow_pickle=False)
    return {**data, "grasp": grasp, "camera_position": (extrinsic @ np.r_[grasp[13:16], 1])[:3]}


def project_grasp(grasp, intrinsic, extrinsic):
    """按 GraspNet 的接近轴、开口宽度和插入深度绘制 U 形夹爪。"""
    width, depth = grasp[1], grasp[3]
    local = np.array([[depth, -width / 2, 0], [-.02, -width / 2, 0],
                      [-.02, width / 2, 0], [depth, width / 2, 0]])
    world = local @ grasp[4:13].reshape(3, 3).T + grasp[13:16]
    camera = world @ extrinsic[:3, :3].T + extrinsic[:3, 3]
    if not np.isfinite(camera).all() or np.any(camera[:, 2] <= 0):
        return None
    pixels = camera @ intrinsic.T
    return pixels[:, :2] / pixels[:, 2:3]


class GraspDebugView:
    def __init__(self):
        import matplotlib.pyplot as plt

        self.plt = plt
        plt.rcParams.update({"font.family": ["Noto Serif CJK SC", "DejaVu Sans"], "axes.unicode_minus": False})
        self.fig, self.axes = plt.subplots(1, 3, figsize=(15, 5), layout="constrained")
        self.fig.canvas.manager.set_window_title("Agent debug: grounding + depth + GraspNet prediction")
        for axis, title in zip(self.axes, ("视觉定位 / Grounding", "深度 / 米", "GraspNet 抓取预测")):
            axis.set_title(title)
            axis.axis("off")
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        self.rgb = self.axes[0].imshow(blank)
        self.depth = self.axes[1].imshow(np.zeros((480, 640)), cmap="gray", vmin=0, vmax=1.4)
        self.prediction = self.axes[2].imshow(blank)
        self.annotations = []
        self.fig.suptitle("等待当前任务观测")
        plt.show(block=False)

    def clear(self):
        for artist in self.annotations:
            artist.remove()
        self.annotations.clear()

    def pump(self):
        if self.is_open():
            self.fig.canvas.flush_events()

    def is_open(self):
        return self.plt.fignum_exists(self.fig.number)

    def frame(self, rgb, depth):
        height, width = rgb.shape[:2]
        for artist, array in ((self.rgb, rgb), (self.depth, depth), (self.prediction, rgb)):
            artist.set_data(array)
            artist.set_extent((-.5, width - .5, height - .5, -.5))
        for axis in self.axes:
            axis.set(xlim=(-.5, width - .5), ylim=(height - .5, -.5))

    def __call__(self, stage, data):
        from matplotlib.patches import Rectangle

        if not self.is_open():
            return
        if stage == "before_step":
            self.pump()
            return
        if stage == "instruction":
            self.clear()
            self.fig.suptitle(data["instruction"] + " · 等待新观测")
        elif stage == "reset":
            self.clear()
            self.frame(data["rgb"], data["depth"])
            self.fig.suptitle("场景已重置 · 等待任务")
        elif stage == "observation":
            self.clear()
            capture = data["directory"] / ("verification" if data.get("verification") else "capture")
            with Image.open(capture / "rgb.png") as image:
                self.frame(np.asarray(image.convert("RGB")), np.load(capture / "depth.npy", allow_pickle=False))
            self.fig.suptitle("验证观测" if data.get("verification") else "规划观测 · 执行时保留本帧")
        elif stage == "plan_ready":
            self.clear()
            height, width = self.rgb.get_array().shape[:2]
            for key, color in (("object", "lime"), ("destination", "orange")):
                target = data["plan"].get(key)
                if target is None:
                    continue
                x1, y1, x2, y2 = np.array(target["bbox"]) * [width, height, width, height] / 1000
                label = ("target: " if key == "object" else "destination: ") + target["color"]
                for axis in (self.axes[0], self.axes[2]):
                    box = Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=color, linewidth=2)
                    axis.add_patch(box)
                    self.annotations.extend([box, axis.text(x1, y1, label, color=color, fontsize=9,
                                                           va="bottom", bbox={"facecolor": "black", "alpha": .7})])
        elif stage == "selected_grasp":
            capture = data["directory"] / "capture"
            pixels = project_grasp(data["grasp"], np.load(capture / "camera_intrinsic.npy", allow_pickle=False),
                                   np.load(capture / "camera_extrinsic.npy", allow_pickle=False))
            if pixels is not None:
                self.annotations.extend(self.axes[2].plot(*pixels.T, color="cyan", linewidth=2.5))
            text = f"score={data['grasp'][0]:.3f}; width={data['grasp'][1]:.3f} m"
            if pixels is None:
                text += " (outside camera)"
            self.annotations.append(self.axes[2].text(.02, .02, text, transform=self.axes[2].transAxes,
                                                     color="yellow", bbox={"facecolor": "black", "alpha": .7}))
            self.fig.savefig(data["directory"] / "grasp-debug.png")
        else:
            return
        self.fig.canvas.draw_idle()
        self.pump()

    def close(self):
        self.plt.close(self.fig)


def _run_window(events):
    from queue import Empty

    view = GraspDebugView()
    try:
        while view.is_open():
            try:
                event = events.get(timeout=.05)
            except Empty:
                view.pump()
                continue
            if event is None:
                break
            view(*event)
    finally:
        view.close()


class DebugWindow:
    """独立 GUI 进程，避免新增窗口与 SAPIEN 的 X11 事件处理冲突。"""

    def __init__(self):
        import multiprocessing

        context = multiprocessing.get_context("spawn")
        self.events = context.Queue()
        self.process = context.Process(target=_run_window, args=(self.events,), daemon=True)
        self.process.start()

    def __call__(self, stage, data):
        if stage not in ("reset", "instruction", "observation", "plan_ready", "selected_grasp"):
            return
        if not self.process.is_alive():
            return
        if stage == "reset":
            sensor = data["obs"]["sensor_data"]["base_camera"]
            data = {"rgb": sensor["rgb"][0].cpu().numpy(),
                    "depth": sensor["depth"][0, ..., 0].cpu().numpy() / 1000}
        self.events.put((stage, data))

    def close(self):
        if self.process.is_alive():
            self.events.put(None)
            self.process.join(timeout=5)
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(timeout=2)
        self.events.cancel_join_thread()
        self.events.close()
