# 五个模块与 V3 持续交互

本轮以 2026-10-13 用户更新的计划为准：先完成全部功能，再集中验收一次。`IMPLEMENTED` 表示入口和最小检查完成，`INTEGRATED` 表示调用方已迁移，`VALIDATED` 只在最终对应验收通过后标记。状态与证据见 [当前进展](progress.md) 和 [测试记录](tests.md)。

## 独立入口与职责

| 模块 | 入口 | 输入与结果 |
| --- | --- | --- |
| VLM Backend | `vlm.plan/locate/verify` | 首次语义解析；恢复时一次请求定位锁定的方块和容器；执行后判断 `success/failure/uncertain`。保留严格 JSON、超时、禁止重定向与密钥保护 |
| Task Planner | `task_planner.plan_task` | 检查动作与身份，锁定恢复任务，检查已有夹持，生成本地固定步骤；不执行验证、IK、抓取或重试 |
| Placement Planner | `placement_planner.locate_destination/plan_place` | 当前 RGB-D 容器定位；根据抓取位姿和视觉夹持偏移生成升高、转正、预放置、放置、释放、撤离六个位姿 |
| Grasp Selector | `grasp_selector.filter_candidates/select_grasp` | am-grasp 做分数、宽度、点云碰撞检查；am-sim 做方向、居中、桌面间隙与完整关键点路径 IK；返回首个可执行候选、路径及淘汰统计 |
| Resource Manager | `resource_manager.ResourceManager` | 串行外部调用、剩余时间、GPU 准入、GraspNet 进程监测与进程组回收；不管理机器人策略 |
| 连续 Demo | `demo.DemoSession/DemoView/main` | 同一底层环境、终端逐条指令、状态保持、重置／退出、双窗口与每任务记录 |

Task Planner 的 `pick` 步骤为 `LOCATE_OBJECT → GRASP → LIFT → HOLD`；`pick_place` 为 `LOCATE_OBJECT → LOCATE_DESTINATION → GRASP → LIFT → PLACE`。V3 控制器追加 `VERIFY`。模型只返回 `status/action/object/destination` 和归一化框，不能插入运动指令。

若已夹持同一物体，新搬运任务只规划 `LOCATE_DESTINATION → PLAN_PLACE → PLACE`，V3 再验证，不调用 GraspNet。夹持其他物体或要求再抓取时在规划阶段拒绝。恢复阶段调用 `locate()`，不重新解释原始指令，不沿用旧框。

Placement Planner 输出 `lift_pose/upright_pose/pre_place_pose/place_pose/release_pose/retreat_pose`。筛选器检查该规划，执行器读取同一组关键点；夹持后续任务用当前图像重新估计偏移。共享模块对 GraspNet 碰撞扩展延迟导入，am-sim 不加载其 CUDA 扩展。

V3 只抓取任务在稳定抬升后采用同一规划器的高处竖直姿态观察，筛选时也检查该段 IK，减少腕部遮住方块导致的验证误判。

筛选统计包含模型分数、宽度和碰撞淘汰数，以及机器人侧方向、居中、桌面间隙、IK 淘汰数和选中排名。机器人侧找到首个可执行候选即结束，不把未检查的剩余候选称为有效候选。

## 状态与恢复

控制器记录 `IDLE/PLANNING/LOCATING/GRASPING/HOLDING/PLACING/VERIFYING/RECOVERING/SUCCESS/STOPPED` 的阶段历史。任务结束后会话回到 `IDLE`，或在仍夹持时保留 `HOLDING`。

每任务最多三轮、180 秒。首次验证 `uncertain` 时，只重新观察、重新定位并再次验证，不盲目重复机械动作；恢复后仍不确定则停止。明确失败才依据 VLM 的 `regrasp/replace/stop` 和夹持安全检查决定后续动作。重新观察同样消耗轮数与时间预算。

碰撞、目标丢失、任务超时、资源失败、GraspNet 崩溃／CUDA OOM、恢复耗尽或最终验证失败进入 `STOPPED`。即使资源故障发生在零控制步时也停止；只接受 `reset` 或 `quit`。初次规划阶段的歧义、非法响应或夹持目标不符，可以修正指令后继续。

真实物体位姿只用于独立评测，不参与定位、路径、恢复或会话状态决策。夹持传感用于安全与继续放置。结果同时记录真值 `success`、模型 `agent_status` 与控制器 `state`，不把它们混为一项。固定场景验收要求真值和模型都成功。

V1 不调用 VLM；V2 不自动重试。V3 每次图文调用最多额外重试一次连接错误，HTTP／协议错误不重试。仿真驻留，远程请求与 GraspNet 串行；退出、超时、取消和异常都会回收本任务的子进程，不干预外部 GPU 负载。

## 显存依据

现有一次真实标定在 RTX 3060 Laptop 6144 MiB 上，同时打开 960×720 SAPIEN、10×7 英寸 Matplotlib，并保持仿真和真实 GraspNet 推理。按子进程 PID 采样的峰值为 **1142 MiB**；加 256 MiB 后，`GRASPNET_MIN_FREE_VRAM = 1398`。

[完整标定记录](validation/modules-resource-results.json) 保留采样与环境依据。本次沿用同一部署和窗口尺寸，不重复标定。此值不是 Torch 保留显存或整卡占用。每次推理记录空闲量与子进程峰值；不足或监控不可用拒绝启动。当前仅支持单 GPU；环境、权重、分辨率或设备变化时重新标定。准入后外部负载增长仍可能导致 OOM。

## 启动

在项目根目录，用现有 am-sim 解释器执行：

```bash
/home/fatbro/software/miniforge3/envs/am-sim/bin/python -m agentic_manipulation.demo
```

默认使用相邻的 am-grasp 解释器、`weights/checkpoint-rs.tar` 和自动创建的 `outputs/demo/session-时间/`。也可显式指定 `--output`、`--grasp-python`、`--checkpoint`、`--seed`；输出目录必须不存在。不会安装环境或下载权重。

```text
Task > 把红色方块放进黄色容器
Task > 把绿色方块放进紫色容器
Task > 把蓝色方块放进黄色容器
Task > reset
Task > 抓起红色方块
Task > 把红色方块放进黄色容器
Task > quit
```

场景只在启动和 `reset` 时初始化。重置保留当前种子，清除夹持、控制器与当前场景任务计数；历史任务目录和会话总序号保留，避免覆盖已有证据。任务使用底层环境，累计步数不会触发单任务回合截断。输入结束、Ctrl-C 或关闭任一窗口也退出并清理。

## 窗口与记录

SAPIEN 以固定视角显示机器人执行，隐藏原生编辑面板以免遮挡场景；Matplotlib 显示 RGB、米制深度、规划观测点云、选中抓取轴、目标框、阶段、恢复原因和资源状态。动作时持续刷新，旧目标框在动作前清除。等待 VLM、GraspNet 和 IK 时保留最近画面并显示等待原因。同步调用期间窗口事件可能延至调用返回处理；Ctrl-C 可中断。

每条任务保存指令、固定步骤、当前观测、点云、候选、筛选统计、抓取姿态、放置关键点、解析后的 VLM 响应、验证、恢复记录、状态历史、资源数据与耗时。重新观察／直接放置的轮次没有新 GraspNet 产物，这是预期行为。

`session.json` 保存任务索引、状态、夹持和重置／中断事件；`task-NNN/` 保存各任务报告与录像。`session.mp4` 保存全会话相机帧，按仿真控制频率编码，不包含网络等待的墙钟时长。用户最新要求优先完成 Demo、不再录像；正式完整展示录像暂缓。功能验收见 [本轮结果索引](validation/modules-results.json)。

仅支持红绿蓝方块、黄紫容器和远程 VLM。暂不加入本地模型、多步指令、任意物体、复杂避障或实机。
