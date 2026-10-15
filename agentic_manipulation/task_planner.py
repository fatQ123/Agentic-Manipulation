"""单目标任务规划：语义身份由模型解析，执行步骤由本地确定。"""

from . import vlm


def task_identity(plan):
    return plan["action"], plan["object"]["color"], (plan.get("destination") or {}).get("color")


def plan_task(instruction, image_path, config, identity=None, held_object=None, timeout=60):
    result = (vlm.plan(instruction, image_path, config, timeout) if identity is None
              else vlm.locate(image_path, identity, config, timeout))
    if identity is not None and task_identity(result) != tuple(identity):
        raise ValueError("replan changed the original task identity")
    if held_object is not None:
        if result["action"] != "pick_place" or result["object"]["color"] != held_object:
            raise ValueError(f"Robot is already holding {held_object} cube. Reset or place the same cube first.")
        steps = ["LOCATE_DESTINATION", "PLAN_PLACE", "PLACE"]
    else:
        steps = (["LOCATE_OBJECT", "GRASP", "LIFT", "HOLD"]
             if result["action"] == "pick" else
             ["LOCATE_OBJECT", "LOCATE_DESTINATION", "GRASP", "LIFT", "PLACE"])
    return {**result, "steps": steps}
