"""API 检测脚本的必要请求与失败检查，不调用真实收费服务。"""

import json
from pathlib import Path
import ssl
import tempfile
import unittest
from unittest.mock import MagicMock, patch
import urllib.error

from PIL import Image

from agentic_manipulation.check_api import check_api


class ApiCheckTests(unittest.TestCase):
    def test_text_image_and_failed_requests(self):
        config = {"VLM_BASE_URL": "https://example.invalid/v1/", "VLM_MODEL": "test",
                  "VLM_API_KEY": "test-secret"}
        response = MagicMock()
        response.__enter__.return_value = response
        response.read.return_value = json.dumps({"choices": [{"finish_reason": "stop",
            "message": {"content": "API_OK test-secret"}}]}).encode()
        with patch("agentic_manipulation.check_api.urllib.request.build_opener") as opener:
            opener.return_value.open.return_value = response
            reply, elapsed = check_api(config)
            self.assertEqual(reply, "API_OK [密钥已隐藏]")
            self.assertGreaterEqual(elapsed, 0)
            request = opener.return_value.open.call_args.args[0]
            self.assertEqual(request.full_url, "https://example.invalid/v1/chat/completions")
            self.assertEqual(request.get_header("Authorization"), "Bearer test-secret")
            self.assertEqual(json.loads(request.data)["messages"][0]["content"], "Reply with exactly API_OK.")
            with tempfile.TemporaryDirectory() as directory:
                image = Path(directory) / "image.png"
                Image.new("RGB", (4, 4)).save(image)
                check_api(config, image)
                request = opener.return_value.open.call_args.args[0]
                self.assertIn("data:image/png;base64,", request.data.decode())
            response.read.return_value = b'{"choices": []}'
            with self.assertRaisesRegex(ValueError, "格式"):
                check_api(config)
            opener.return_value.open.side_effect = urllib.error.HTTPError(
                request.full_url, 401, "test-secret", {}, None)
            with self.assertRaisesRegex(RuntimeError, "HTTP 401") as error:
                check_api(config)
            self.assertNotIn("test-secret", str(error.exception))
            opener.return_value.open.side_effect = urllib.error.URLError(ssl.SSLError("test-secret"))
            with self.assertRaisesRegex(RuntimeError, "TLS"):
                check_api(config)
            with self.assertRaisesRegex(ValueError, "正数"):
                check_api(config, timeout=float("nan"))
