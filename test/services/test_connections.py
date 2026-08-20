import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import requests

from app.services import connections


class TestResultShape(unittest.TestCase):
    def test_status_reads_as_connected_or_error(self):
        self.assertEqual(
            connections.ConnectionResult("x", True, "ok").status, "connected"
        )
        self.assertEqual(
            connections.ConnectionResult("x", False, "bad").status, "error"
        )

    def test_failure_messages_name_the_exception_type(self):
        """"连接失败"没有价值，用户需要知道是鉴权、网络还是配置问题。"""
        result = connections._timed(
            "x", MagicMock(side_effect=PermissionError("key rejected"))
        )
        self.assertFalse(result.ok)
        self.assertIn("PermissionError", result.message)
        self.assertIn("key rejected", result.message)

    def test_long_provider_errors_are_truncated(self):
        """某些 SDK 会把整个请求体塞进异常里，原样显示会撑爆页面。"""
        result = connections._timed("x", MagicMock(side_effect=ValueError("y" * 900)))
        self.assertLess(len(result.message), 400)
        self.assertTrue(result.message.endswith("…"))

    def test_success_records_a_latency(self):
        result = connections._timed("x", lambda: "fine")
        self.assertTrue(result.ok)
        self.assertEqual(result.message, "fine")
        self.assertGreaterEqual(result.latency_ms, 0)


class TestLlmConnection(unittest.TestCase):
    def test_builds_an_isolated_config_snapshot(self):
        """探测不能改动全局配置：用户可能只是在试一把还没决定保存的 Key。"""
        from app.config import config

        before = dict(config.app)
        snapshot = connections.build_llm_config(
            "openai", api_key="sk-probe", model_name="gpt-probe"
        )
        self.assertEqual(snapshot["llm_provider"], "openai")
        self.assertEqual(snapshot["openai_api_key"], "sk-probe")
        self.assertEqual(snapshot["openai_model_name"], "gpt-probe")
        self.assertEqual(dict(config.app), before)

    def test_blank_fields_fall_back_to_saved_config(self):
        """"测试当前配置"和"测试一把新 Key"应当共用同一条路径。"""
        with patch.dict("app.config.config.app", {"openai_api_key": "saved"}):
            snapshot = connections.build_llm_config("openai")
        self.assertEqual(snapshot["openai_api_key"], "saved")

    def test_unknown_provider_is_rejected(self):
        with self.assertRaises(ValueError):
            connections.build_llm_config("not-a-provider")

    def test_reports_connected_when_the_model_replies(self):
        with patch("app.services.llm._generate_response", return_value="OK"):
            result = connections.test_llm("openai", api_key="k", model_name="gpt-4o")
        self.assertTrue(result.ok)
        self.assertIn("gpt-4o", result.message)

    def test_an_empty_reply_counts_as_a_failure(self):
        """返回空串通常意味着额度耗尽或模型名无效，不能算连接成功。"""
        with patch("app.services.llm._generate_response", return_value=""):
            result = connections.test_llm("openai", api_key="k")
        self.assertFalse(result.ok)

    def test_an_error_string_reply_is_a_failure_not_a_success(self):
        """
        _generate_response 把失败编码成 "Error: ..." 字符串而不是抛异常。
        不识别这个前缀，一把无效的 Key 会被报成"连接成功"。
        """
        with patch(
            "app.services.llm._generate_response",
            return_value="Error: moonshot: api_key is not set",
        ):
            result = connections.test_llm("moonshot")
        self.assertFalse(result.ok)
        self.assertIn("api_key is not set", result.message)
        self.assertNotIn("Error: Error:", result.message)

    def test_provider_errors_are_surfaced_verbatim(self):
        with patch(
            "app.services.llm._generate_response",
            side_effect=ValueError("insufficient balance"),
        ):
            result = connections.test_llm("openai", api_key="k")
        self.assertFalse(result.ok)
        self.assertIn("insufficient balance", result.message)

    def test_qwen_is_selectable_like_any_other_provider(self):
        snapshot = connections.build_llm_config("qwen", api_key="sk-qwen")
        self.assertEqual(snapshot["llm_provider"], "qwen")
        self.assertEqual(snapshot["qwen_api_key"], "sk-qwen")

    def test_ollama_needs_no_api_key(self):
        self.assertIn("ollama", connections.llm_provider_ids())
        snapshot = connections.build_llm_config("ollama", base_url="http://localhost:11434")
        self.assertEqual(snapshot["ollama_base_url"], "http://localhost:11434")


class TestMotionConnection(unittest.TestCase):
    """按秒计费的服务不能靠真的生成一次来测试。"""

    def _response(self, status: int, body=None):
        response = MagicMock()
        response.status_code = status
        response.json.return_value = body or {}
        return response

    def test_a_not_found_task_proves_the_key_authenticated(self):
        with patch.dict(
            "app.config.config.app", {"kling_api_key": "k"}
        ), patch.object(requests, "get", return_value=self._response(404)):
            result = connections.test_motion_provider()
        self.assertTrue(result.ok)
        self.assertIn("authenticated", result.message)

    def test_a_rejected_key_reports_the_provider_reason(self):
        with patch.dict("app.config.config.app", {"kling_api_key": "bad"}), patch.object(
            requests,
            "get",
            return_value=self._response(401, {"message": "invalid api key"}),
        ):
            result = connections.test_motion_provider()
        self.assertFalse(result.ok)
        self.assertIn("invalid api key", result.message)

    def test_a_missing_key_says_so_instead_of_calling_out(self):
        with patch.dict("app.config.config.app", {"kling_api_key": ""}), patch.object(
            requests, "get"
        ) as get:
            result = connections.test_motion_provider()
        get.assert_not_called()
        self.assertFalse(result.ok)
        self.assertIn("kling_api_key", result.message)

    def test_a_provider_outage_is_distinguished_from_a_bad_key(self):
        with patch.dict("app.config.config.app", {"kling_api_key": "k"}), patch.object(
            requests, "get", return_value=self._response(503)
        ):
            result = connections.test_motion_provider()
        self.assertFalse(result.ok)
        self.assertIn("unavailable", result.message)

    def test_network_failures_are_reported_as_such(self):
        with patch.dict("app.config.config.app", {"kling_api_key": "k"}), patch.object(
            requests, "get", side_effect=requests.ConnectionError("dns failure")
        ):
            result = connections.test_motion_provider()
        self.assertFalse(result.ok)
        self.assertIn("dns failure", result.message)


class TestStillConnection(unittest.TestCase):
    def test_lists_models_rather_than_generating_an_image(self):
        """生成一张就计一次费，列模型足以证明 Key 有效。"""
        client = MagicMock()
        client.__enter__ = lambda self_: client
        client.__exit__ = lambda *args: False
        client.models.list.return_value = [
            SimpleNamespace(name="models/gemini-3-pro-image-preview")
        ]
        with patch("google.genai.Client", return_value=client):
            result = connections.test_still_provider(api_key="k")
        self.assertTrue(result.ok)
        client.models.list.assert_called_once()

    def test_a_missing_key_is_reported_before_any_call(self):
        with patch.dict(
            "app.config.config.app", {"gemini_api_key": "", "gemini_image_api_key": ""}
        ):
            result = connections.test_still_provider()
        self.assertFalse(result.ok)
        self.assertIn("gemini_api_key", result.message)


class TestStockConnection(unittest.TestCase):
    def test_rate_limiting_is_not_reported_as_a_bad_key(self):
        """429 说明 Key 是好的，只是被限流，与鉴权失败是两回事。"""
        response = MagicMock()
        response.status_code = 429
        with patch.object(requests, "get", return_value=response):
            result = connections.test_stock_provider("pexels", api_key="k")
        self.assertFalse(result.ok)
        self.assertIn("throttled", result.message)

    def test_a_working_key_reports_connected(self):
        response = MagicMock()
        response.status_code = 200
        with patch.object(requests, "get", return_value=response):
            result = connections.test_stock_provider("pixabay", api_key="k")
        self.assertTrue(result.ok)

    def test_unknown_providers_are_rejected_by_name(self):
        result = connections.test_stock_provider("shutterstock")
        self.assertFalse(result.ok)
        self.assertIn("unknown stock provider", result.message)


class TestTestAll(unittest.TestCase):
    def test_unconfigured_providers_are_skipped_not_failed(self):
        """把没配置的供应商全报成失败，会让真正的错误淹没在一片红色里。"""
        with patch.dict(
            "app.config.config.app",
            {
                "gemini_api_key": "",
                "gemini_image_api_key": "",
                "kling_api_key": "",
                "sonilo_api_key": "",
                "pexels_api_keys": [],
                "pixabay_api_keys": [],
            },
        ), patch.dict("app.config.config.elevenlabs", {"api_key": ""}), patch.object(
            connections, "test_llm", return_value=connections.ConnectionResult("llm", True, "ok")
        ):
            results = connections.test_all()
        self.assertEqual([entry.name for entry in results], ["llm"])

    def test_configured_providers_are_included(self):
        with patch.dict(
            "app.config.config.app", {"kling_api_key": "k"}
        ), patch.object(
            connections, "test_llm", return_value=connections.ConnectionResult("llm", True, "ok")
        ), patch.object(
            connections,
            "test_motion_provider",
            return_value=connections.ConnectionResult("motion:kling", True, "ok"),
        ):
            names = [entry.name for entry in connections.test_all()]
        self.assertIn("motion:kling", names)


if __name__ == "__main__":
    unittest.main()
