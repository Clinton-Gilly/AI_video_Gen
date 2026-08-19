"""
Kling 图生视频。

按秒计费，是单集成本的主要来源，因此调用方应当先经过
``app.services.visual.planner`` 决定哪些镜头值得动，而不是逐镜调用本模块。

接口形态是"提交任务 → 轮询 → 下载产物"。响应解析刻意写得宽松：官方接口把
结果包在 ``data`` 里，各家网关（Segmind、CometAPI、AI/ML API 等）会把同一份
结果拍平或换字段名。只认一种形状会让换网关变成改代码，因此这里按多个常见
位置查找，取到第一个可用的即可。
"""

from __future__ import annotations

import base64
import time
from pathlib import Path
from typing import Any, Mapping

import requests
from loguru import logger

from app.config import config
from app.services.visual.base import (
    MAX_CLIP_BYTES,
    VisualConfigurationError,
    VisualGenerationError,
    download_to_file,
    validate_aspect,
)


DEFAULT_BASE_URL = "https://api.klingai.com"
IMAGE2VIDEO_PATH = "/v1/videos/image2video"
DEFAULT_MODEL = "kling-v2.6-pro"
# standard 比 professional 便宜，画质对竖屏短剧足够。默认选便宜的那一档。
DEFAULT_MODE = "std"

DEFAULT_POLL_INTERVAL_SECONDS = 5.0
DEFAULT_TIMEOUT_SECONDS = 600.0
# 供应商按秒计费且各档位支持的时长有限，这里只放行常见档位，避免把一个会被
# 服务端拒绝的时长提交上去、白等一轮超时。
SUPPORTED_DURATIONS = (5, 10)

_TERMINAL_SUCCESS = {"succeed", "succeeded", "success", "completed"}
_TERMINAL_FAILURE = {"failed", "failure", "error", "cancelled", "canceled"}


class KlingVideoProvider:
    """把静帧驱动成短视频。"""

    name = "kling"

    def __init__(self, app_config: Any = None):
        self._config = app_config if app_config is not None else config.app

    def _api_key(self) -> str:
        return str(self._config.get("kling_api_key") or "").strip()

    def _base_url(self) -> str:
        return str(
            self._config.get("kling_base_url") or DEFAULT_BASE_URL
        ).strip().rstrip("/")

    def _model(self) -> str:
        return str(self._config.get("kling_model") or "").strip() or DEFAULT_MODEL

    def _mode(self) -> str:
        return str(self._config.get("kling_mode") or "").strip() or DEFAULT_MODE

    def _timeout_seconds(self) -> float:
        # 显式判空而不是用 or：0 是合法取值，用 or 会把它当成"未配置"而悄悄
        # 回退到默认的十分钟，让一个本该立刻放弃的调用空等十分钟。
        value = self._config.get("kling_timeout_seconds")
        return float(DEFAULT_TIMEOUT_SECONDS if value in (None, "") else value)

    def _poll_interval(self) -> float:
        value = self._config.get("kling_poll_interval_seconds")
        return float(DEFAULT_POLL_INTERVAL_SECONDS if value in (None, "") else value)

    def is_enabled(self) -> bool:
        return bool(self._api_key())

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self._api_key()}",
            "Content-Type": "application/json",
        }

    def animate(
        self,
        *,
        still_path: str,
        prompt: str,
        duration: float,
        output_path: str,
        aspect: str = "9:16",
    ) -> str:
        if not self._api_key():
            raise VisualConfigurationError(
                "Kling animation requires kling_api_key in config.toml"
            )
        aspect = validate_aspect(aspect)
        source = Path(still_path)
        if not source.is_file():
            raise VisualGenerationError(f"still image not found: {still_path}")

        billed_duration = _billable_duration(duration)
        payload = {
            "model": self._model(),
            "mode": self._mode(),
            "prompt": prompt,
            "duration": billed_duration,
            "aspect_ratio": aspect,
            # 静帧在本地生成，没有公网地址可给，因此内联上传。
            "image": base64.b64encode(source.read_bytes()).decode("ascii"),
        }

        logger.info(
            f"submitting animation: model={payload['model']}, "
            f"mode={payload['mode']}, duration={billed_duration}s, aspect={aspect}"
        )
        task_id = self._submit(payload)
        video_url = self._poll(task_id)
        return download_to_file(video_url, output_path, MAX_CLIP_BYTES)

    def _submit(self, payload: Mapping[str, Any]) -> str:
        url = f"{self._base_url()}{IMAGE2VIDEO_PATH}"
        try:
            response = requests.post(
                url, json=payload, headers=self._headers(), timeout=(10, 120)
            )
        except requests.RequestException as exc:
            raise VisualGenerationError(f"failed to reach Kling: {exc}") from exc

        body = _json_body(response)
        if response.status_code not in (200, 201):
            raise VisualGenerationError(
                f"Kling rejected the request: HTTP {response.status_code} "
                f"{_error_message(body)}"
            )
        _raise_for_business_error(body)

        task_id = _first_value(body, ("task_id", "taskId", "id"))
        if not task_id:
            raise VisualGenerationError("Kling did not return a task id")
        logger.info(f"animation task submitted: task_id={task_id}")
        return str(task_id)

    def _poll(self, task_id: str) -> str:
        url = f"{self._base_url()}{IMAGE2VIDEO_PATH}/{task_id}"
        deadline = time.monotonic() + self._timeout_seconds()
        interval = self._poll_interval()

        while True:
            try:
                response = requests.get(url, headers=self._headers(), timeout=(10, 60))
            except requests.RequestException as exc:
                raise VisualGenerationError(
                    f"failed to poll Kling task {task_id}: {exc}"
                ) from exc

            body = _json_body(response)
            if response.status_code != 200:
                raise VisualGenerationError(
                    f"failed to poll Kling task {task_id}: "
                    f"HTTP {response.status_code} {_error_message(body)}"
                )
            _raise_for_business_error(body)

            status = str(
                _first_value(body, ("task_status", "taskStatus", "status")) or ""
            ).lower()
            if status in _TERMINAL_FAILURE:
                detail = (
                    _first_value(body, ("task_status_msg", "taskStatusMsg", "message"))
                    or status
                )
                raise VisualGenerationError(f"Kling task {task_id} failed: {detail}")

            if status in _TERMINAL_SUCCESS:
                video_url = _extract_video_url(body)
                if not video_url:
                    raise VisualGenerationError(
                        f"Kling task {task_id} succeeded but returned no video url"
                    )
                return video_url

            if time.monotonic() >= deadline:
                raise VisualGenerationError(
                    f"Kling task {task_id} did not finish within "
                    f"{self._timeout_seconds():.0f}s (last status: {status or 'unknown'})"
                )
            time.sleep(interval)


def _billable_duration(duration: float) -> int:
    """把镜头时长向上对齐到供应商支持的计费档位。"""
    for candidate in SUPPORTED_DURATIONS:
        if duration <= candidate:
            return candidate
    return SUPPORTED_DURATIONS[-1]


def _json_body(response: requests.Response) -> dict:
    try:
        body = response.json()
    except ValueError:
        return {}
    return body if isinstance(body, dict) else {}


def _unwrap(body: Mapping[str, Any]) -> Mapping[str, Any]:
    """官方接口把结果包在 ``data`` 里，网关通常拍平，两种都要能读。"""
    data = body.get("data")
    return data if isinstance(data, Mapping) else body


def _first_value(body: Mapping[str, Any], keys: tuple[str, ...]) -> Any:
    for source in (_unwrap(body), body):
        for key in keys:
            value = source.get(key)
            if value not in (None, ""):
                return value
    return None


def _error_message(body: Mapping[str, Any]) -> str:
    return str(_first_value(body, ("message", "msg", "error")) or "").strip()


def _raise_for_business_error(body: Mapping[str, Any]) -> None:
    """
    HTTP 200 也可能携带业务错误码。

    Kling 用 ``code`` 表示业务状态，0 为成功。网关不一定透传该字段，因此只在
    出现且非 0 时报错，避免把没有该字段的正常响应误判成失败。
    """
    code = body.get("code")
    if code in (None, 0, "0"):
        return
    raise VisualGenerationError(
        f"Kling returned error code {code}: {_error_message(body) or 'unknown error'}"
    )


def _extract_video_url(body: Mapping[str, Any]) -> str:
    """
    在多种响应形状里找出成片地址。

    官方形状是 ``data.task_result.videos[0].url``；各网关会换成 ``video_url``、
    ``output``、``url`` 或一个字符串数组。逐一尝试比要求用户按网关改代码好。
    """
    data = _unwrap(body)

    result = data.get("task_result")
    if isinstance(result, Mapping):
        videos = result.get("videos")
        if isinstance(videos, list) and videos:
            first = videos[0]
            if isinstance(first, Mapping):
                url = first.get("url") or first.get("video_url")
                if url:
                    return str(url)
            elif isinstance(first, str):
                return first

    for key in ("video_url", "videoUrl", "url", "output", "output_url"):
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, list) and value and isinstance(value[0], str):
            return value[0]

    return ""
