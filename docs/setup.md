# 手动环境准备

`Agent.md` 明确要求：**Conda 操作、环境配置和环境内依赖下载由用户手动执行。** 本文只提供命令，不提供自动安装脚本。助手可在已有解释器中运行项目及测试，但不得代为创建环境、修改环境或安装依赖。

## 先判断你是否需要安装

当前开发电脑的 am-sim 与 am-grasp、CUDA 扩展、源码和权重均已准备并通过实际运行。**本机直接按 [新手指南](getting-started.md) 启动，不要重复安装。**

新电脑需要手动完成下列步骤，再继续 [GraspNet 准备](graspnet-setup.md)。实测平台为 Linux Mint 22.3、RTX 3060 Laptop 6 GB；尚未在全新机器从零复现。两个 [包清单](am-sim-packages.txt) 是实测记录，不是跨平台锁定文件。

命令中的项目路径应替换为你的实际路径。安装遇到错误时先保留报错，不要继续下一组。

## 先只准备 am-sim

请在你自己的终端执行，避免影响已有学习环境：

```bash
conda create -n am-sim python=3.11
conda activate am-sim
python -m pip install "mani_skill==3.0.1" "sapien==3.0.3" torch numpy scipy pillow matplotlib imageio imageio-ffmpeg
```

以上是最初的准备步骤；实际组合现已完成采集验收，包清单是本机观察记录，不是跨机器兼容性保证。采用单场景 CPU 物理仿真和 GPU 渲染，不要求此环境安装 CUDA Toolkit。

官方安装与接口依据：

- [ManiSkill 安装](https://maniskill.readthedocs.io/en/latest/user_guide/getting_started/installation.html)
- [RGB-D 与相机参数](https://maniskill.readthedocs.io/en/latest/user_guide/concepts/observation.html)
- [ManiSkill 3.0.1 环境接口](https://github.com/haosulab/ManiSkill/blob/v3.0.1/mani_skill/envs/sapien_env.py)

然后在项目目录运行：

```bash
cd /home/fatbro/Agentproject/codex_Agentic-Manipulation
python -m agentic_manipulation.capture --output outputs/capture-0 --seed 0
python -m agentic_manipulation.pointcloud outputs/capture-0 --output outputs/cloud-0
```

`capture` 在环境初始化前拒绝不完整依赖，不执行安装。若 ManiSkill 提示缺少模型／场景资产，请保留报错，由用户依据官方说明手动准备，不绕过下载确认。默认 Panda／PickCube 使用官方内置资源。

安装完成后告知助手解释器的绝对路径（通常为 `/home/fatbro/software/miniforge3/envs/am-sim/bin/python`），即可在不执行 Conda 命令的前提下继续验收。

## 点云查看（可选，后于相机采集）

只导出 NumPy 点云无需 Open3D。需要三维窗口时，由用户在准备好的环境执行：

```bash
python -m pip install open3d
python -m agentic_manipulation.pointcloud outputs/capture-0 --output outputs/cloud-view --view
```

## 下一步：GraspNet 独立基础环境（仅用户手动执行）

本机已完成以下步骤；新机器先准备基础环境，再阅读 [GraspNet 准备](graspnet-setup.md)。

相机链路已通过，可以开始准备 am-grasp。先准备独立 Python 和 GPU PyTorch；以下组合在本机已通过推理，跨机器仍需检查扩展编译和运行。本步骤不修改已跑通的 am-sim。

```bash
conda create -n am-grasp python=3.10
conda activate am-grasp
python -m pip install "numpy==1.26.4" scipy pillow open3d ninja
python -m pip install "torch==2.2.2" --index-url https://download.pytorch.org/whl/cu118
```

选择旧一些的独立 PyTorch 以减少旧 CUDA 扩展接口变动；该 cu118 安装版本来自 [PyTorch 官方历史版本说明](https://pytorch.org/get-started/previous-versions/)。NumPy 固定 1.x，避免旧抓取代码遇到 NumPy 2 的接口变化。

准备完成后提供 `/home/fatbro/software/miniforge3/envs/am-grasp/bin/python`，先检查 GPU 张量运行，再决定工具链安装步骤。

**还未包含的后续工作：** CUDA Toolkit／nvcc、匹配的 C++ 编译器、官方 GraspNet 源码、PointNet2／KNN 扩展、graspnetAPI 和官方权重。新机器须检查 nvcc 和 C++ 编译器是否匹配 CUDA 11.8。编译工具链、扩展及权重的下载／安装同样需要用户手动操作，命令见 GraspNet 准备说明。

官方依据：[GraspNet baseline](https://github.com/graspnet/graspnet-baseline)、[PointNet2 构建入口](https://github.com/graspnet/graspnet-baseline/blob/main/pointnet2/setup.py)、[KNN 构建入口](https://github.com/graspnet/graspnet-baseline/blob/main/knn/setup.py)。

不要把官方原始 PyTorch 1.6 依赖混入 am-sim，也不要把显卡驱动显示的 CUDA 版本当作已安装 CUDA Toolkit。
