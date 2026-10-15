# 文档索引

每次新会话先读本页及 [当前进展](progress.md)，再开展工作。所有日期采用北京时间。

| 文档 | 用途 |
| --- | --- |
| [Agent.md](../Agent.md) | 用户规定的实施约束，优先遵守 |
| [README](../README.md) | 项目入口与运行命令 |
| [新手使用指南](getting-started.md) | 前置知识、准备、配置、逐条操作与排错 |
| [发布准备状态](release.md) | 清理范围、保留文件与正式发布前事项 |
| [连续任务完整报告](validation/modules-session-details.json) | 清理原始输出前保留的连续会话、夹持续放与断连报告 |
| [五模块与持续演示](modules-demo.md) | 模块职责、接口、资源标定和连续交互 |
| [本轮验收结果索引](validation/modules-results.json) | 五模块、持续会话、复查与正式录像暂缓状态 |
| [本轮 V3 十场景](validation/modules-v3-results.json) | 新实现一次集中验收 10/10，后续针对性修复见测试记录 |
| [显存标定结果](validation/modules-resource-results.json) | 带窗口 GraspNet 进程峰值与准入预算 |
| [工作内容记录](worklog.md) | 实际修改与处理结果 |
| [测试记录](tests.md) | 执行过的检查、结果及未验证事项 |
| [当前推进进展](progress.md) | 已完成、待办与下一步 |
| [问题记录](issues.md) | 未解决问题与已解决问题摘要 |
| [开发与发布计划](plan.md) | 三个大版本的边界与验收门槛 |
| [V2 完整验收结果](validation/v2-results.json) | 固定十组自然语言搬运的实际原始汇总，10/10 |
| [V2 演示录像](validation/v2-demo.mp4) | seed 1 蓝色方块放入紫色碗全过程 |
| [V3 完整验收结果](validation/v3-results.json) | 同 V2 固定十组场景，最终 10/10 |
| [V3 只抓取结果](validation/v3-pick-results.json) | 蓝色方块抓起、稳定夹持与实际 VLM 成功判断 |
| [V3 真实恢复结果](validation/v3-recovery-results.json) | 抓空、未释放和目标丢失三类真实服务验收，3/3 |
| [V3 恢复演示](validation/v3-demo.mp4) | 首次抓空后重新观察，第二轮抓取并放置成功 |
| [V3 运行说明](v3.md) | 视觉反馈、有限恢复、独立判分与已完成的本地验收 |
| [V2 运行说明](v2.md) | 本地模型配置、语言搬运入口、验收及限制 |
| [手动环境准备](setup.md) | 仅供用户执行的安装说明 |
| [am-sim 包清单](am-sim-packages.txt) | 真实 RGB-D 采集通过后的环境记录，不是自动安装脚本 |
| [GraspNet 手动准备](graspnet-setup.md) | 基础环境通过后的工具链、源码、扩展与权重准备 |
| [V1 完整验收结果](validation/v1-results.json) | 固定十场景原始汇总，包含两次失败 |
| [V1 演示录像](validation/v1-demo.mp4) | seed 1 真实 RGB-D 视觉抓取与抬升 |
| [am-grasp 包清单](am-grasp-packages.txt) | 官方 GraspNet 与项目推理通过后的包记录，包含扩展 |

实现代码位于 `agentic_manipulation/`，最少必要测试位于 `tests/`。运行产物自动写入已忽略的 `outputs/`；2026-10-13 已按用户要求删除历史临时输出，精选证据保留在 `docs/validation/`。

发布仓库：[fatQ123/Agentic-Manipulation](https://github.com/fatQ123/Agentic-Manipulation)。发布结果以远程 main 和版本标签为准。

[独立 API 检测](api-check.md)：不启动仿真，检测文本或指定图片调用。

## 2026-10-15 GitHub 发布

本次 GitHub main 发布范围及本地历史备份策略见 [发布记录](release.md)；本次发布检查见 [测试记录](tests.md)。
