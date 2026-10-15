# 独立 API 检测

从项目根目录运行。复用 `vlm.configuration()`，读取 `.env` 的 `VLM_BASE_URL`、`VLM_MODEL`、`VLM_API_KEY`；进程环境变量优先。无需安装新依赖，不启动仿真或执行机器人动作。

```bash
cd /home/fatbro/Agentproject/codex_Agentic-Manipulation
/home/fatbro/software/miniforge3/envs/am-sim/bin/python -m agentic_manipulation.check_api

# 可选：检查图片输入，换成你自己的 PNG/JPEG 路径。
/home/fatbro/software/miniforge3/envs/am-sim/bin/python -m agentic_manipulation.check_api --image /绝对路径/图片.png
```

默认发送一条简短文本并打印模型回答、耗时；`--image` 发送指定图片并要求一句中文描述。地址追加 `/chat/completions`；成功须返回完整的非空文本，退出 0。失败退出 1，提示 HTTP 状态或 TLS／超时／连接错误，隐藏密钥及错误正文。不自动重试，可用 `--timeout 120` 调整默认 60 秒的网络超时。

文本成功只说明文本调用成功；模型是否能看图需运行图片检测并检查描述。TLS／网络线路间歇失败时，单次成功不代表长时间稳定。每次真实运行会按服务商规则产生 API 用量。

2026-10-12：离线请求／错误检查与命令行帮助通过；本次开发没有调用真实服务。
