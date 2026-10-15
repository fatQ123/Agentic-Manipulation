"""独立检测 OpenAI 兼容 API；不启动仿真，不执行机器人动作。"""

import argparse
import base64
import http.client
import json
import math
from pathlib import Path
import socket
import ssl
import time
import urllib.error
import urllib.request

from PIL import Image

from .vlm import configuration, NoRedirect


def check_api(config, image=None, timeout=60):
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("超时时间必须是有限的正数")
    content = "Reply with exactly API_OK."
    if image is not None:
        with Image.open(image) as picture:
            if picture.format not in {"PNG", "JPEG"}:
                raise ValueError("检测图片只支持 PNG 或 JPEG")
            mime = "image/png" if picture.format == "PNG" else "image/jpeg"
            picture.load()
        content = [
            {"type": "text", "text": "请用一句中文描述这张图片。"},
            {"type": "image_url", "image_url": {
                "url": f"data:{mime};base64," + base64.b64encode(Path(image).read_bytes()).decode("ascii")
            }},
        ]
    request = urllib.request.Request(
        config["VLM_BASE_URL"].rstrip("/") + "/chat/completions",
        data=json.dumps({"model": config["VLM_MODEL"], "messages": [
            {"role": "user", "content": content}
        ]}).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": "Bearer " + config["VLM_API_KEY"]},
    )
    start = time.monotonic()
    try:
        with urllib.request.build_opener(NoRedirect).open(request, timeout=timeout) as response:
            body = response.read(1_000_001)
    except urllib.error.HTTPError as exc:
        hint = {401: "检查 API 密钥", 403: "检查账号或模型权限", 404: "检查基础地址和模型名",
                429: "检查额度或调用频率"}.get(exc.code, "检查服务状态和接口配置")
        raise RuntimeError(f"HTTP {exc.code}：{hint}（错误正文已隐藏）") from None
    except (urllib.error.URLError, TimeoutError, OSError, http.client.HTTPException) as exc:
        reason = exc.reason if isinstance(exc, urllib.error.URLError) else exc
        if isinstance(reason, ssl.SSLError):
            hint = "TLS 连接失败，请检查服务线路、代理或证书"
        elif isinstance(reason, (TimeoutError, socket.timeout)):
            hint = "请求超时，请检查网络、服务响应时间或增大 --timeout"
        else:
            hint = "连接失败，请检查网络、代理和基础地址"
        raise RuntimeError(hint) from None
    if len(body) > 1_000_000:
        raise ValueError("响应超过 1 MB，检测失败")
    try:
        choice = json.loads(body)["choices"][0]
        reply = choice["message"]["content"]
        if not isinstance(reply, str) or not reply.strip() or choice.get("finish_reason") != "stop":
            raise ValueError("模型没有返回完整的文本回答")
    except (KeyError, IndexError, TypeError, UnicodeError, json.JSONDecodeError):
        raise ValueError("响应不符合 Chat Completions 格式") from None
    return reply.replace(config["VLM_API_KEY"], "[密钥已隐藏]"), time.monotonic() - start


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, help="可选 PNG/JPEG 图片，用于检查视觉输入")
    parser.add_argument("--timeout", type=float, default=60, help="请求超时秒数，默认 60")
    args = parser.parse_args()
    try:
        reply, elapsed = check_api(configuration(), args.image, args.timeout)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(1, f"API 检测失败：{exc}\n")
    print(f"API 调用成功（{'图文' if args.image else '文本'}），耗时 {elapsed:.2f} 秒")
    print(f"模型回答：{reply}")


if __name__ == "__main__":
    main()
