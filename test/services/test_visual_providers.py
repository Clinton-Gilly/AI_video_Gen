import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import requests

from app.services import visual
from app.services.visual import base
from app.services.visual.kling_video import KlingVideoProvider, _extract_video_url


class TestBaseHelpers(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_write_bytes_creates_missing_parent_directories(self):
        target = self.root / "nested" / "deeper" / "still.png"
        base.write_bytes(str(target), b"payload", base.MAX_STILL_BYTES)
        self.assertEqual(target.read_bytes(), b"payload")

    def test_write_bytes_rejects_an_empty_payload(self):
        with self.assertRaises(base.VisualGenerationError):
            base.write_bytes(str(self.root / "a.png"), b"", base.MAX_STILL_BYTES)

    def test_write_bytes_rejects_an_oversized_payload(self):
        with self.assertRaises(base.VisualGenerationError):
            base.write_bytes(str(self.root / "a.png"), b"12345", max_bytes=4)

    def test_write_bytes_leaves_no_temp_file_behind_on_failure(self):
        """中断后残留的半个文件会被后续步骤当成有效素材。"""
        with self.assertRaises(base.VisualGenerationError):
            base.write_bytes(str(self.root / "a.png"), b"12345", max_bytes=4)
        self.assertEqual(list(self.root.iterdir()), [])

    def test_unsupported_aspect_is_rejected(self):
        with self.assertRaises(base.VisualGenerationError):
            base.validate_aspect("4:3")

    def test_download_aborts_once_the_size_limit_is_passed(self):
        """整份读完再判断大小会让异常响应直接占满内存。"""
        response = MagicMock()
        response.status_code = 200
        response.iter_content.return_value = [b"x" * 100] * 50
        response.__enter__ = lambda self_: response
        response.__exit__ = lambda *args: False

        with patch.object(requests, "get", return_value=response):
            with self.assertRaises(base.VisualGenerationError):
                base.download_to_file(
                    "https://example.com/v.mp4",
                    str(self.root / "v.mp4"),
                    max_bytes=1000,
                )
        self.assertEqual(list(self.root.iterdir()), [])

    def test_download_rejects_an_empty_url(self):
        with self.assertRaises(base.VisualGenerationError):
            base.download_to_file("", str(self.root / "v.mp4"), max_bytes=100)


class TestKlingResponseShapes(unittest.TestCase):
    """官方接口和各家网关的响应形状不同，只认一种会让换网关变成改代码。"""

    def test_reads_the_official_nested_shape(self):
        body = {
            "data": {
                "task_result": {"videos": [{"url": "https://cdn/official.mp4"}]}
            }
        }
        self.assertEqual(_extract_video_url(body), "https://cdn/official.mp4")

    def test_reads_a_flattened_gateway_shape(self):
        self.assertEqual(
            _extract_video_url({"video_url": "https://cdn/gateway.mp4"}),
            "https://cdn/gateway.mp4",
        )

    def test_reads_a_string_array_result(self):
        self.assertEqual(
            _extract_video_url({"output": ["https://cdn/array.mp4"]}),
            "https://cdn/array.mp4",
        )

    def test_returns_empty_when_no_url_is_present(self):
        self.assertEqual(_extract_video_url({"data": {"task_status": "processing"}}), "")


class TestKlingProvider(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.still = self.root / "shot.png"
        self.still.write_bytes(b"fake-png")
        self.provider = KlingVideoProvider(
            {"kling_api_key": "k", "kling_poll_interval_seconds": 0}
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def _response(self, body, status=200):
        response = MagicMock()
        response.status_code = status
        response.json.return_value = body
        return response

    def test_is_disabled_without_an_api_key(self):
        self.assertFalse(KlingVideoProvider({}).is_enabled())
        self.assertTrue(self.provider.is_enabled())

    def test_animate_without_a_key_raises_a_configuration_error(self):
        with self.assertRaises(base.VisualConfigurationError):
            KlingVideoProvider({}).animate(
                still_path=str(self.still),
                prompt="slow push in",
                duration=3.0,
                output_path=str(self.root / "out.mp4"),
            )

    def test_missing_still_is_rejected_before_any_request(self):
        with patch.object(requests, "post") as post:
            with self.assertRaises(base.VisualGenerationError):
                self.provider.animate(
                    still_path=str(self.root / "nope.png"),
                    prompt="p",
                    duration=3.0,
                    output_path=str(self.root / "out.mp4"),
                )
        post.assert_not_called()

    def test_shot_duration_is_rounded_up_to_a_billable_tier(self):
        """提交一个服务端会拒绝的时长只会白等一轮超时。"""
        post = self._response({"data": {"task_id": "t1"}})
        get = self._response(
            {
                "data": {
                    "task_status": "succeed",
                    "task_result": {"videos": [{"url": "https://cdn/v.mp4"}]},
                }
            }
        )
        with patch.object(requests, "post", return_value=post) as post_mock, patch.object(
            requests, "get", return_value=get
        ), patch.object(base, "download_to_file", return_value="out.mp4"), patch(
            "app.services.visual.kling_video.download_to_file", return_value="out.mp4"
        ):
            self.provider.animate(
                still_path=str(self.still),
                prompt="p",
                duration=3.0,
                output_path=str(self.root / "out.mp4"),
            )
        self.assertEqual(post_mock.call_args.kwargs["json"]["duration"], 5)

    def test_a_failed_task_surfaces_the_provider_message(self):
        post = self._response({"data": {"task_id": "t1"}})
        get = self._response(
            {"data": {"task_status": "failed", "task_status_msg": "content rejected"}}
        )
        with patch.object(requests, "post", return_value=post), patch.object(
            requests, "get", return_value=get
        ):
            with self.assertRaises(base.VisualGenerationError) as context:
                self.provider.animate(
                    still_path=str(self.still),
                    prompt="p",
                    duration=3.0,
                    output_path=str(self.root / "out.mp4"),
                )
        self.assertIn("content rejected", str(context.exception))

    def test_a_business_error_code_on_http_200_is_not_treated_as_success(self):
        post = self._response({"code": 1101, "message": "insufficient balance"})
        with patch.object(requests, "post", return_value=post):
            with self.assertRaises(base.VisualGenerationError) as context:
                self.provider.animate(
                    still_path=str(self.still),
                    prompt="p",
                    duration=3.0,
                    output_path=str(self.root / "out.mp4"),
                )
        self.assertIn("insufficient balance", str(context.exception))

    def test_polling_stops_at_the_timeout_instead_of_looping_forever(self):
        post = self._response({"data": {"task_id": "t1"}})
        get = self._response({"data": {"task_status": "processing"}})
        provider = KlingVideoProvider(
            {
                "kling_api_key": "k",
                "kling_poll_interval_seconds": 0,
                "kling_timeout_seconds": 0,
            }
        )
        with patch.object(requests, "post", return_value=post), patch.object(
            requests, "get", return_value=get
        ):
            with self.assertRaises(base.VisualGenerationError) as context:
                provider.animate(
                    still_path=str(self.still),
                    prompt="p",
                    duration=3.0,
                    output_path=str(self.root / "out.mp4"),
                )
        self.assertIn("did not finish", str(context.exception))


class TestGeminiImageProvider(unittest.TestCase):
    def test_reuses_the_llm_gemini_key(self):
        """同一个账号不应当要求用户配置两次。"""
        provider = visual.get_still_provider(
            "gemini", app_config={"gemini_api_key": "shared"}
        )
        self.assertTrue(provider.is_enabled())

    def test_is_disabled_without_any_key(self):
        self.assertFalse(visual.get_still_provider("gemini", app_config={}).is_enabled())

    def test_a_text_only_response_reports_what_the_model_said(self):
        """只回文字通常是被安全策略拦了，吞掉理由会让用户无从排查。"""
        from app.services.visual.gemini_images import _extract_image_bytes

        response = SimpleNamespace(
            candidates=[
                SimpleNamespace(
                    content=SimpleNamespace(
                        parts=[SimpleNamespace(inline_data=None, text="blocked: policy")]
                    )
                )
            ]
        )
        with self.assertRaises(base.VisualGenerationError) as context:
            _extract_image_bytes(response)
        self.assertIn("blocked: policy", str(context.exception))

    def test_extracts_inline_image_data(self):
        from app.services.visual.gemini_images import _extract_image_bytes

        response = SimpleNamespace(
            candidates=[
                SimpleNamespace(
                    content=SimpleNamespace(
                        parts=[
                            SimpleNamespace(inline_data=None, text="here you go"),
                            SimpleNamespace(
                                inline_data=SimpleNamespace(data=b"png-bytes"),
                                text=None,
                            ),
                        ]
                    )
                )
            ]
        )
        self.assertEqual(_extract_image_bytes(response), b"png-bytes")


class TestProviderRegistry(unittest.TestCase):
    def test_unknown_providers_are_rejected_by_name(self):
        with self.assertRaises(base.VisualConfigurationError):
            visual.get_still_provider("midjourney", app_config={})
        with self.assertRaises(base.VisualConfigurationError):
            visual.get_motion_provider("sora", app_config={})

    def test_config_selects_the_backend(self):
        self.assertEqual(
            visual.get_motion_provider(app_config={"motion_provider": "kling"}).name,
            "kling",
        )


if __name__ == "__main__":
    unittest.main()
