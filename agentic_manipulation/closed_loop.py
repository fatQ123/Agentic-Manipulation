"""V3 单任务闭环，可在独立场景或持续演示会话中运行。"""

import json
from pathlib import Path
import time

import numpy as np

from .capture import make_env, save_observation
from .pick import PickController, ExecutionFailure, execute_pick, execute_place, infer_grasps
from .placement_planner import locate_destination
from .pointcloud import reconstruct
from .resource_manager import ResourceManager
from .task_planner import plan_task, task_identity
from .vlm import configuration, verify


MAX_ROUNDS = 3
TASK_TIMEOUT = 180.0


def score_task(control, plan):
    """连续一秒独立真值判分；返回值不用于规划或重试决策。"""
    actor = control.target_actor
    pose = control.pose()
    gripping = bool(control.agent.is_grasping(actor))
    success = True
    for _ in range(control.base.control_freq):
        control.step(control.action(pose, -1 if gripping else 1))
        cube = actor.pose.to_transformation_matrix()[0].cpu().numpy()
        extent = control.base.cube_half_size * np.abs(cube[:3, :3]).sum(axis=1)
        bottom = cube[2, 3] - extent[2]
        held = bool(control.agent.is_grasping(actor))
        if plan["action"] == "pick":
            success &= bottom >= 0.05 and held
        else:
            bowl = control.base.bowls[plan["destination"]["color"]].pose.p[0].cpu().numpy()
            inside = np.all(np.abs(cube[:2, 3] - bowl[:2]) + extent[:2] <= 0.07)
            speed = np.linalg.norm(actor.linear_velocity[0].cpu().numpy())
            spin = np.linalg.norm(actor.angular_velocity[0].cpu().numpy())
            success &= inside and 0.005 < bottom < 0.02 and speed <= 0.02 and spin <= 0.2 and not held
    return {"success": bool(success), "hold_s": 1.0, "cube_position": cube[:3, 3].tolist()}


def run_closed_loop(output, grasp_python, checkpoint, seed, instruction, env=None, notify=None, observation_pose=None):
    import imageio.v2 as imageio

    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    resources = ResourceManager(start + TASK_TIMEOUT, notify)
    report = {"seed": seed, "version": "v3", "instruction": instruction, "success": False,
              "agent_status": "stopped", "max_rounds": MAX_ROUNDS, "timeout_s": TASK_TIMEOUT, "rounds": [],
              "state": "IDLE", "state_history": [], "held_object": None}
    own_env = env is None
    writer = control = None

    def emit(stage, data=None):
        if notify is not None:
            notify(stage, data or {})

    def set_state(state, reason=""):
        report["state"] = state
        report["state_history"].append({"state": state, "elapsed_s": time.monotonic() - start, "reason": reason})
        emit("state", {"state": state, "reason": reason})

    try:
        with resources:
            config = configuration()
            report["model"] = config["VLM_MODEL"]
            if own_env:
                env = make_env(task="LanguageTabletop-v2", max_episode_steps=2400)
                env.reset(seed=seed)
            writer = imageio.get_writer(output / "pick.mp4", fps=env.unwrapped.control_freq)
            control = PickController(env, writer, deadline=resources.deadline, notify=notify)
            if observation_pose is None:
                observation_pose = control.pose()
            held_colors = [color for color, actor in env.unwrapped.cubes.items()
                           if bool(control.agent.is_grasping(actor))]
            if len(held_colors) > 1:
                raise RuntimeError("multiple held objects; reset required")
            report["held_object"] = held_colors[0] if held_colors else None
            recovery = "regrasp"
            for number in range(1, MAX_ROUNDS + 1):
                resources.remaining(TASK_TIMEOUT)
                directory = output / f"round-{number}"
                directory.mkdir()
                trial = {"round": number, "recovery": recovery, "phases": []}
                report["rounds"].append(trial)
                set_state("PLANNING" if number == 1 else "RECOVERING")
                emit("observe", {"round": number})
                save_observation(env.unwrapped.get_obs(), directory / "capture", seed, "LanguageTabletop-v2")
                emit("observation", {"directory": directory})
                identity = None if number == 1 else task_identity(report["plan"])
                plan = resources.call_vlm(plan_task, instruction, directory / "capture" / "rgb.png", config,
                                          identity=identity, held_object=report["held_object"] if number == 1 else None,
                                          trial=trial, stage="plan" if number == 1 else "locate", retry=True)
                plan["steps"] = plan["steps"] + ["VERIFY"]
                trial["plan"] = plan
                (directory / "plan.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
                emit("plan_ready", {"directory": directory, "plan": plan})
                if number == 1:
                    report["plan"] = plan
                    if held_colors:
                        recovery = "replace"
                        trial["recovery"] = recovery
                    control.target_actor = env.unwrapped.cubes[plan["object"]["color"]]
                    control.target_name = control.target_actor._objs[0].name
                destination = None
                set_state("LOCATING")
                if plan["action"] == "pick_place":
                    destination = locate_destination(directory / "capture", plan["destination"])
                    trial["destination_world"] = destination.tolist()
                try:
                    if recovery == "reobserve":
                        trial["phases"].append("reobserve_only")
                    elif recovery == "replace":
                        if destination is None or not bool(control.agent.is_grasping(control.target_actor)):
                            raise RuntimeError("unsafe replacement: no held target or destination")
                        reconstruct(directory / "capture", directory / "cloud", plan["object"])
                        with np.load(directory / "cloud" / "points.npz", allow_pickle=False) as cloud:
                            center = (cloud["target_world"].min(axis=0) + cloud["target_world"].max(axis=0)) / 2
                        target = control.pose()
                        emit("cloud_ready", {"directory": directory})
                        set_state("PLACING")
                        with resources.phase("execute"):
                            execute_place(control, target, center, destination, plan["destination"]["color"], trial)
                    else:
                        set_state("GRASPING")
                        candidates = infer_grasps(directory, grasp_python, checkpoint, seed + number - 1, plan, resources)
                        emit("cloud_ready", {"directory": directory})
                        with resources.phase("execute"):
                            target, center = execute_pick(control, directory, candidates, destination, trial, seed,
                                                          observe_held=destination is None)
                            set_state("HOLDING")
                            if destination is not None:
                                set_state("PLACING")
                                execute_place(control, target, center, destination, plan["destination"]["color"], trial)
                except ExecutionFailure as exc:
                    trial["execution_failure"] = str(exc)
                held = bool(control.agent.is_grasping(control.target_actor))
                report["held_object"] = plan["object"]["color"] if held else None
                safe = control.pose()
                safe[2, 3] = max(safe[2, 3], 0.25)
                control.move(safe, -1 if held else 1)
                if held and "observation_pose" in trial:
                    # 复用已通过 IK 的高处转正姿态，让方块露出手腕，供视觉独立验证。
                    control.move(np.asarray(trial["observation_pose"]), -1)
                if not held:
                    control.move(observation_pose, 1)
                trial["truth"] = score_task(control, plan)
                save_observation(env.unwrapped.get_obs(), directory / "verification", seed, "LanguageTabletop-v2")
                emit("observation", {"directory": directory, "verification": True})
                set_state("VERIFYING")
                verdict = resources.call_vlm(verify, instruction, directory / "verification" / "rgb.png", report["plan"], config,
                                             trial=trial, stage="verification", retry=True)
                resources.remaining(TASK_TIMEOUT)
                trial["verdict"] = verdict
                trial["steps"] = control.steps
                (directory / "result.json").write_text(json.dumps(trial, indent=2) + "\n", encoding="utf-8")
                report["success"] = trial["truth"]["success"]
                report["agent_status"] = verdict["status"]
                emit("verdict", {"verdict": verdict, "round": number})
                if verdict["status"] == "uncertain" and number == 1:
                    recovery = "reobserve"
                    set_state("RECOVERING", verdict["reason"])
                    continue
                if verdict["status"] != "failure" or verdict["recovery"] == "stop":
                    report["stop_reason"] = verdict["reason"]
                    set_state("SUCCESS" if verdict["status"] == "success" else "STOPPED", verdict["reason"])
                    break
                if number == MAX_ROUNDS:
                    report["stop_reason"] = "execution round limit reached"
                    set_state("STOPPED", report["stop_reason"])
                    break
                recovery = verdict["recovery"]
                set_state("RECOVERING", verdict["reason"])
                if recovery == "regrasp" and held:
                    raise RuntimeError("unsafe regrasp: target is still held")
    except KeyboardInterrupt:
        report.update(success=False, agent_status="stopped", failure="task cancelled")
        report["state"] = "STOPPED"
        raise
    except Exception as exc:
        report.update(success=False, agent_status="stopped", failure=str(exc))
        rejected = isinstance(exc, ValueError) and report["state"] == "PLANNING" and control is not None and control.steps == 0
        report["state"] = ("HOLDING" if report["held_object"] else "IDLE") if rejected else "STOPPED"
        report["state_history"].append({"state": report["state"], "elapsed_s": time.monotonic() - start, "reason": str(exc)})
    finally:
        resources.close()
        report["resources"] = resources.report
        if control is not None:
            report["steps"] = control.steps
        try:
            if writer is not None:
                writer.close()
        finally:
            if own_env and env is not None:
                env.close()
            report["elapsed_s"] = time.monotonic() - start
            verdicts = [r for r in report["rounds"] if "verdict" in r]
            report["vlm_judgments"] = {
                "total": len(verdicts),
                "false_success": sum(r["verdict"]["status"] == "success" and not r["truth"]["success"] for r in verdicts),
                "false_failure": sum(r["verdict"]["status"] == "failure" and r["truth"]["success"] for r in verdicts),
                "uncertain": sum(r["verdict"]["status"] == "uncertain" for r in verdicts),
            }
            (output / "result.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
