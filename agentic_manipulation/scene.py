"""V2 固定相机、多色方块与两个有碰撞几何的开口碗。"""

import numpy as np
import sapien
import torch
from mani_skill.envs.tasks.tabletop.pick_cube import PickCubeEnv
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils


@register_env("LanguageTabletop-v2", max_episode_steps=1000)
class LanguageTabletopEnv(PickCubeEnv):
    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=[0.4, 0.4, 0.65], target=[0, 0, 0])
        return [CameraConfig("base_camera", pose, 640, 480, 1.2, 0.01, 100)]

    def _load_scene(self, options):
        super()._load_scene(options)
        self.cubes = {"red": self.cube}
        for color, rgba in (("green", [0, 0.8, 0, 1]), ("blue", [0, 0.1, 1, 1])):
            self.cubes[color] = actors.build_cube(self.scene, half_size=self.cube_half_size, color=rgba, name=color + "_cube", initial_pose=sapien.Pose(p=[0, 0, 0.02]))
        self.bowls = {}
        for color, rgba in (("yellow", [1, 0.85, 0, 1]), ("purple", [0.65, 0, 0.85, 1])):
            builder = self.scene.create_actor_builder()
            material = sapien.render.RenderMaterial(base_color=rgba)
            builder.add_box_collision(half_size=[0.07, 0.07, 0.005])
            builder.add_box_visual(half_size=[0.07, 0.07, 0.005], material=material)
            for x, y, hx, hy in ((-0.075, 0, 0.005, 0.08), (0.075, 0, 0.005, 0.08), (0, -0.075, 0.07, 0.005), (0, 0.075, 0.07, 0.005)):
                pose = sapien.Pose(p=[x, y, 0.02])
                builder.add_box_collision(pose=pose, half_size=[hx, hy, 0.025])
                builder.add_box_visual(pose=pose, half_size=[hx, hy, 0.025], material=material)
            builder.initial_pose = sapien.Pose(p=[0, 0, 0.005])
            self.bowls[color] = builder.build_kinematic(name=color + "_bowl")

    def _initialize_episode(self, env_idx, options):
        super()._initialize_episode(env_idx, options)
        # 观察姿态将手臂收回，避免初始夹爪遮住待抓方块。
        qpos = self.agent.robot.get_qpos().clone()
        qpos[:, 1] = -0.35
        self.agent.reset(qpos)
        with torch.device(self.device):
            slots = torch.tensor([[-0.12, -0.16], [-0.12, 0], [-0.12, 0.16]])
            slots = slots[torch.randperm(3)]
            for index, cube in enumerate(self.cubes.values()):
                xy = slots[index] + (torch.rand(2) - 0.5) * 0.025
                angle = torch.rand(1).item() * np.pi
                cube.set_pose(Pose.create_from_pq(p=torch.cat((xy, torch.tensor([self.cube_half_size])))[None], q=torch.tensor([[np.cos(angle / 2), 0, 0, np.sin(angle / 2)]])))
            slots = torch.tensor([[0.08, -0.13], [0.08, 0.13]])[torch.randperm(2)]
            for index, bowl in enumerate(self.bowls.values()):
                xy = slots[index] + (torch.rand(2) - 0.5) * 0.015
                bowl.set_pose(Pose.create_from_pq(p=torch.cat((xy, torch.tensor([0.005])))[None]))
