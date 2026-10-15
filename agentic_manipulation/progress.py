"""将执行事件转换为即时终端进度，不输出每一帧。"""

import numpy as np


class TerminalProgress:
    def __init__(self):
        self.round = 1
        self.object = "target"
        self.destination = "destination"

    def __call__(self, stage, data):
        prefix = f"[步骤 1][轮次 {self.round}]"
        message = None
        if stage == "instruction":
            self.round = 1
            self.object, self.destination = "target", "destination"
            message = "[agent-client] 开始新任务，等待当前场景执行：" + data["instruction"]
        elif stage == "observe":
            self.round = data["round"]
            message = f"[观察] capturing RGB-D frame (round {self.round})"
        elif stage in ("plan", "locate"):
            message = prefix + f"[识别] grounding attempt {self.round}"
        elif stage == "plan_ready":
            plan = data["plan"]
            self.object = plan["object"]["color"] + "_cube"
            destination = plan.get("destination")
            self.destination = destination["color"] + "_bowl" if destination else ""
            description = f"pick {self.object}" if not destination else f"move {self.object} to {self.destination}"
            message = (f"[场景] 视觉定位：{self.object}, bbox={plan['object']['bbox']}"
                       + (f"; {self.destination}, bbox={destination['bbox']}" if destination else "")
                       + ("\n[规划] planned 1 task" if self.round == 1 else "")
                       + f"\n{prefix}[开始] {description}")
        elif stage == "grasp_inference":
            message = prefix + f"[抓取位姿] GraspNet for {self.object}"
        elif stage == "selected_grasp":
            camera = np.round(np.asarray(data["camera_position"], dtype=float), 6).tolist()
            world = np.round(np.asarray(data["target"][:3, 3], dtype=float), 6).tolist()
            grasp = data["grasp"]
            message = (prefix + f"[抓取位姿] selected grasp camera position {camera}; world EE position {world}; "
                       f"score={grasp[0]:.4f}, width={grasp[1]:.4f} m")
        elif stage == "raise":
            message = prefix + "[执行] atomic pick"
        elif stage == "place":
            message = prefix + f"[执行] atomic place into {self.destination}"
        elif stage == "state" and data["state"] == "RECOVERING" and data.get("reason"):
            message = prefix + "[重试] " + data["reason"]
        elif stage == "verification":
            message = prefix + "[验证] 检查执行后的场景"
        elif stage == "verdict":
            message = prefix + f"[验证] {data['verdict']['status']}: {data['verdict']['reason']}"
        elif stage == "task_complete":
            report = data["report"]
            message = (f"[结束] {report['state']}; success={report['success']}; "
                       + report.get("failure", report.get("stop_reason", "")))
        if message is not None:
            print(message, flush=True)
