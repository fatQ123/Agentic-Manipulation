# GraspNet 准备（新机器由用户手动执行）

## 已经通过，不用重建环境

解释器：`/home/fatbro/software/miniforge3/envs/am-grasp/bin/python`。

已检查 Python 3.10.22、PyTorch 2.2.2+cu118、NumPy 1.26.4、Open3D 0.20.0。RTX 3060 的 GPU 矩阵运算通过，Open3D 成功读取真实目标点云的 348 个点。完整清单见 [am-grasp 包清单](am-grasp-packages.txt)。

当前电脑这些内容已准备完成；下面用于说明新机器需要的 CUDA Toolkit／nvcc、C++ 编译器、GraspNet 源码、PointNet2／KNN 扩展、graspnetAPI 和权重。PyTorch 自带 CUDA 运行库，**能做 GPU 运算并不代表能编译扩展**。

以下全是供用户在自己的终端执行的说明。遵守 `Agent.md`，助手没有执行这些安装或下载命令。各组按顺序执行；遇到错误就停在该组，保留完整报错，不继续执行后续组。

## 1. CUDA Toolkit 和 C++ 编译器

```bash
conda activate am-grasp
conda install -c nvidia/label/cuda-11.8.0 cuda-toolkit=11.8.0
sudo apt install gcc-11 g++-11
```

Toolkit 安装在当前独立环境，不需要重装显卡驱动。系统已有 G++ 13.3，不能直接用它编译这套 CUDA 11.8 扩展；gcc-11／g++-11 是 Ubuntu Noble 仓库中可用的独立版本，不切换系统默认编译器。

在**同一个终端**设置本次构建所需变量：

```bash
export CUDA_HOME="$CONDA_PREFIX"
export CC=/usr/bin/gcc-11
export CXX=/usr/bin/g++-11
export TORCH_CUDA_ARCH_LIST="8.6"
export MAX_JOBS=2

"$CUDA_HOME/bin/nvcc" --version
"$CXX" --version
python -c "import torch; from torch.utils.cpp_extension import CUDA_HOME; print('torch:', torch.__version__); print('CUDA:', torch.version.cuda); print('CUDA_HOME:', CUDA_HOME); print('GPU:', torch.cuda.is_available())"
```

检查点：nvcc 为 11.8；C++ 编译器为 11.x；Torch 保持 2.2.2+cu118；CUDA_HOME 指向 am-grasp；GPU 为 True。若打开新终端，需要重新激活环境并设置上述变量，不要写入全局配置。

依据：[NVIDIA CUDA 11.8 安装说明](https://docs.nvidia.com/cuda/archive/11.8.0/cuda-installation-guide-linux/index.html#conda-installation)、[NVIDIA cuda-toolkit 包记录](https://api.anaconda.org/package/nvidia/cuda-toolkit)、[Ubuntu g++-11](https://packages.ubuntu.com/noble/g%2B%2B-11)。此组合已在本机编译运行，新机器仍须验证；Linux Mint 22.3 不是 CUDA 11.8 文档中的原始认证发行版。

## 2. 下载官方源码和补齐 Python 依赖

```bash
cd /home/fatbro/Agentproject/codex_Agentic-Manipulation
mkdir -p third_party weights
git clone https://github.com/graspnet/graspnet-baseline.git third_party/graspnet-baseline
git clone https://github.com/graspnet/graspnetAPI.git third_party/graspnetAPI

python -m pip install "setuptools<70" wheel tqdm tensorboard
python -m pip install "numpy==1.26.4" ./third_party/graspnetAPI
```

如果目录已经存在，不要删除或覆盖，告知现有路径。此处源码目录和权重目录已被本项目 Git 忽略。克隆完成后会记录上游提交版本；不把整个依赖仓库上传到项目。

安装 graspnetAPI 时同时保留 NumPy 1.26.4 约束，让解析器选择可用依赖，避免升级到 NumPy 2。若解析失败，提供报错，不用 `--no-deps` 跳过缺失依赖，也不升级本项目的 am-sim。

## 3. 编译并安装两个 CUDA 扩展

仍在上述配置好的终端、项目根目录执行：

```bash
python -m pip install --no-build-isolation --no-deps ./third_party/graspnet-baseline/pointnet2
python -m pip install --no-build-isolation --no-deps ./third_party/graspnet-baseline/knn
python -c "import pointnet2._ext; import knn_pytorch.knn_pytorch; import graspnetAPI; print('GraspNet extension imports: OK')"
```

这里两个扩展的 `--no-deps` 只用于防止重装已核对的 PyTorch，`--no-build-isolation` 让构建使用当前 GPU PyTorch；这与上一组跳过 graspnetAPI 依赖不同。扩展能导入后，仍需实际运行官方推理才能确认其 CUDA 运算正确。

若发生编译错误，保留第一段实际编译错误及末尾输出；不要加入 `--allow-unsupported-compiler` 或随意替换 CUDA／Torch。不要将本机成功编译理解为所有平台都兼容。

## 4. 下载官方权重

打开 [GraspNet 官方 README 的权重下载部分](https://github.com/graspnet/graspnet-baseline#training-and-testing)，手动下载 RealSense 权重 **checkpoint-rs.tar**，放到：

```text
/home/fatbro/Agentproject/codex_Agentic-Manipulation/weights/checkpoint-rs.tar
```

这是 PyTorch checkpoint，保留原文件，不当作普通 tar 包解压。只使用官方链接提供的权重；不需要下载完整训练数据集。源码附带官方 demo 样例。

## 完成后

告诉助手完成到哪一组；如果全部完成，提供权重路径即可。接下来由助手检查上游版本、运行官方样例推理并接入本项目点云，不会再替你安装包。

2026-10-10 实测：用户已完成准备，两个扩展、官方样例及项目推理通过。真实物理抓取已成功，十场景验收见测试记录；不需要重新安装。

本次源码版本：baseline `280c215129f759ed8649cb4e89fc5dfee55f4f80`；graspnetAPI `eb57dd2092d8dbe05312a29c3d0c22f3226efbfc`。官方 demo 含一个未使用的数据集导入，会间接引用已删除的 `torch._six`；仅删除该导入，没有添加兼容层。补丁保存在 `patches/graspnet-demo.patch`。如重建相同源码，用户可在项目根目录应用：

```bash
git -C third_party/graspnet-baseline apply ../../patches/graspnet-demo.patch
```

实际推理环境包版本见 `am-grasp-packages.txt`。这里只记录已运行的环境，尚未在全新环境从零复现。

上游依据：[GraspNet baseline](https://github.com/graspnet/graspnet-baseline)、[PointNet2 构建文件](https://github.com/graspnet/graspnet-baseline/blob/main/pointnet2/setup.py)、[KNN 构建文件](https://github.com/graspnet/graspnet-baseline/blob/main/knn/setup.py)、[graspnetAPI 依赖](https://github.com/graspnet/graspnetAPI/blob/master/setup.py)。
