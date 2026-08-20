"""
统一的服务连通性测试。

页面上每个供应商都需要回答同一个问题："这把 Key 现在能用吗？"，而各家的
探测方式完全不同——有的要发一次对话请求，有的查订阅信息，有的只能靠一次
被拒绝的查询来反推鉴权是否通过。把差异收敛在这里，调用方（WebUI、API、
CLI）只面对一种结果结构。

两条硬性约束：

* **探测不能花钱。** 图像和视频模型按次/按秒计费，因此这里一律不触发生成，
  只做鉴权和元数据查询。
* **失败必须说清原因。** "连接失败"没有价值；用户需要知道是 Key 没填、
  Key 无效、余额不足，还是网络到不了，才知道下一步做什么。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

from loguru import logger

from app.config import config
from app.models.llm_provider import LLM_PROVIDER_REGISTRY, get_llm_provider


@dataclass(frozen=True)
class ConnectionResult:
    """一次连通性测试的结果。"""

    name: str
    ok: bool
    message: str
    latency_ms: int = 0
    details: dict = field(default_factory=dict)

    @property
    def status(self) -> str:
        return "connected" if self.ok else "error"


def _summarize(exc: BaseException) -> str:
    """
    把异常压成一句可读的原因。

    这段文本会直接显示给用户，因此保留异常类型（区分网络错误和鉴权错误）
    但截断过长的堆栈式消息——某些 SDK 会把整个请求体塞进异常里。
    """
    text = str(exc).strip() or exc.__class__.__name__
    collapsed = " ".join(text.split())
    if len(collapsed) > 300:
        collapsed = collapsed[:300] + "…"
    return f"{type(exc).__name__}: {collapsed}"


def _timed(name: str, probe: Callable[[], str]) -> ConnectionResult:
    """执行一次探测并计时，把任何异常转换成失败结果。"""
    started = time.perf_counter()
    try:
        message = probe() or "connected"
        elapsed = int((time.perf_counter() - started) * 1000)
        logger.info(f"connection ok: {name} ({elapsed}ms)")
        return ConnectionResult(name=name, ok=True, message=message, latency_ms=elapsed)
    except Exception as exc:
        elapsed = int((time.perf_counter() - started) * 1000)
        reason = _summarize(exc)
        logger.warning(f"connection failed: {name} — {reason}")
        return ConnectionResult(
            name=name, ok=False, message=reason, latency_ms=elapsed
        )


# --------------------------------------------------------------------------
# LLM
# --------------------------------------------------------------------------


def llm_provider_ids() -> list[str]:
    return [spec.provider_id for spec in LLM_PROVIDER_REGISTRY]


def build_llm_config(
    provider_id: str,
    api_key: str = "",
    model_name: str = "",
    base_url: str = "",
    extra: Mapping[str, Any] | None = None,
) -> dict:
    """
    组装一份只用于本次探测的配置快照。

    不写回全局配置：用户可能只是想试一把还没决定要不要保存的 Key，探测本身
    不应该改变正在运行的任务所使用的供应商。留空的字段回退到已保存的配置，
    这样"测试当前配置"和"测试一把新 Key"共用同一条路径。
    """
    spec = get_llm_provider(provider_id)
    if spec is None:
        raise ValueError(f"unknown llm provider: {provider_id}")

    snapshot = dict(config.app)
    snapshot["llm_provider"] = spec.provider_id
    if api_key:
        snapshot[spec.config_key("api_key")] = api_key
    if model_name:
        snapshot[spec.config_key("model_name")] = model_name
    if base_url:
        snapshot[spec.config_key("base_url")] = base_url
    for suffix, value in (extra or {}).items():
        if value:
            snapshot[spec.config_key(suffix)] = value
    return snapshot


def test_llm(
    provider_id: str = "",
    api_key: str = "",
    model_name: str = "",
    base_url: str = "",
    extra: Mapping[str, Any] | None = None,
) -> ConnectionResult:
    """
    测试某个 LLM 供应商。

    发一次极短的对话请求：这是唯一能同时验证 Key、Base URL、模型名和账户
    余额的方式，而查模型列表只能验证前两项。请求短到成本可以忽略。
    """
    from app.services import llm

    resolved = provider_id or str(config.app.get("llm_provider", "")).strip()
    if not resolved:
        return ConnectionResult(
            name="llm", ok=False, message="no llm provider selected"
        )

    def probe() -> str:
        snapshot = build_llm_config(
            resolved, api_key, model_name, base_url, extra
        )
        reply = llm._generate_response(
            prompt="Reply with exactly: OK", app_config=snapshot
        )
        text = str(reply or "").strip()
        if not text:
            raise RuntimeError("provider returned an empty response")
        # _generate_response 不抛异常，它把失败原因编码成 "Error: ..." 字符串
        # 返回。不识别这个前缀，一把无效的 Key 会被报成"连接成功"——正好和
        # 这个页面存在的意义相反。
        if text.startswith("Error:"):
            raise RuntimeError(text.removeprefix("Error:").strip())
        spec = get_llm_provider(resolved)
        used = spec.resolve_model_name(
            snapshot.get(spec.config_key("model_name"), "")
        )
        return f"replied via {used}"

    return _timed(f"llm:{resolved}", probe)


# --------------------------------------------------------------------------
# 画面：静帧与图生视频
# --------------------------------------------------------------------------


def test_still_provider(api_key: str = "", model: str = "") -> ConnectionResult:
    """
    测试静帧后端。

    只列模型，不生成图片——生成一张就要计一次费，而列模型足以证明 Key 有效。
    """
    from app.services.visual.gemini_images import DEFAULT_MODEL

    def probe() -> str:
        from google import genai

        key = (
            api_key
            or str(config.app.get("gemini_image_api_key") or "").strip()
            or str(config.app.get("gemini_api_key") or "").strip()
        )
        if not key:
            raise ValueError("gemini_api_key is not set")
        target = model or str(
            config.app.get("gemini_image_model") or DEFAULT_MODEL
        )
        with genai.Client(api_key=key) as client:
            names = [getattr(m, "name", "") for m in client.models.list()]
        if not names:
            raise RuntimeError("no models are visible to this key")
        # 目标模型不在列表里不算失败：不同账号可见的模型集合不同，而 Key
        # 本身已经验证通过。这里只提示，避免误报成连接失败。
        suffix = "" if any(target in name for name in names) else f"; '{target}' not listed"
        return f"{len(names)} models visible{suffix}"

    return _timed("stills:gemini", probe)


def test_motion_provider(api_key: str = "", base_url: str = "") -> ConnectionResult:
    """
    测试图生视频后端。

    不能真的提交一次生成——那按秒计费。改为查询一个不存在的任务：鉴权失败会
    返回 401/403，而鉴权通过、仅任务不存在会返回 404 或业务级 "not found"。
    因此"查不到任务"恰恰证明这把 Key 是有效的。
    """
    import requests

    from app.services.visual.kling_video import (
        DEFAULT_BASE_URL,
        IMAGE2VIDEO_PATH,
        _error_message,
        _json_body,
    )

    def probe() -> str:
        key = api_key or str(config.app.get("kling_api_key") or "").strip()
        if not key:
            raise ValueError("kling_api_key is not set")
        root = (
            base_url
            or str(config.app.get("kling_base_url") or DEFAULT_BASE_URL)
        ).strip().rstrip("/")

        response = requests.get(
            f"{root}{IMAGE2VIDEO_PATH}/connection-probe-does-not-exist",
            headers={"Authorization": f"Bearer {key}"},
            timeout=(10, 30),
        )
        body = _json_body(response)
        if response.status_code in (401, 403):
            raise PermissionError(
                _error_message(body) or "the API key was rejected"
            )
        if response.status_code >= 500:
            raise RuntimeError(
                f"provider is unavailable (HTTP {response.status_code})"
            )
        # 200/404 都说明请求通过了鉴权，只是任务不存在。
        return "authenticated"

    return _timed("motion:kling", probe)


# --------------------------------------------------------------------------
# 配乐
# --------------------------------------------------------------------------


def test_music_provider(provider: str) -> ConnectionResult:
    """测试视频配乐供应商。两家都提供了不计费的账户查询接口。"""
    normalized = (provider or "").strip().lower()

    if normalized == "sonilo":
        from app.services import sonilo

        def probe() -> str:
            if not sonilo.is_enabled():
                raise ValueError("sonilo_api_key is not set")
            payload = sonilo.test_connection()
            return str(payload.get("message") or "account reachable")

        return _timed("music:sonilo", probe)

    if normalized == "elevenlabs":
        from app.services import elevenlabs_music

        def probe() -> str:
            if not elevenlabs_music.is_enabled():
                raise ValueError("elevenlabs api key is not set")
            payload = elevenlabs_music.test_connection()
            return str(payload.get("message") or "account reachable")

        return _timed("music:elevenlabs", probe)

    return ConnectionResult(
        name=f"music:{normalized or 'unknown'}",
        ok=False,
        message=f"unknown music provider: {provider}",
    )


# --------------------------------------------------------------------------
# 素材库
# --------------------------------------------------------------------------


_STOCK_PROBES = {
    "pexels": ("https://api.pexels.com/v1/collections?per_page=1", "api_key"),
    "pixabay": ("https://pixabay.com/api/videos/?per_page=3", "query"),
}


def test_stock_provider(provider: str, api_key: str = "") -> ConnectionResult:
    """测试素材库 Key。两家都有轻量的只读查询接口。"""
    import requests

    normalized = (provider or "").strip().lower()
    probe_spec = _STOCK_PROBES.get(normalized)
    if probe_spec is None:
        return ConnectionResult(
            name=f"stock:{normalized or 'unknown'}",
            ok=False,
            message=f"unknown stock provider: {provider}",
        )
    url, auth_style = probe_spec

    def probe() -> str:
        keys = config.app.get(f"{normalized}_api_keys") or []
        if isinstance(keys, str):
            keys = [keys]
        key = api_key or (str(keys[0]).strip() if keys else "")
        if not key:
            raise ValueError(f"{normalized}_api_keys is not set")

        if auth_style == "api_key":
            response = requests.get(url, headers={"Authorization": key}, timeout=(10, 30))
        else:
            response = requests.get(f"{url}&key={key}", timeout=(10, 30))

        if response.status_code in (401, 403):
            raise PermissionError("the API key was rejected")
        if response.status_code == 429:
            raise RuntimeError("rate limited; the key works but is throttled")
        if response.status_code != 200:
            raise RuntimeError(f"HTTP {response.status_code}")
        return "authenticated"

    return _timed(f"stock:{normalized}", probe)


# --------------------------------------------------------------------------
# 汇总
# --------------------------------------------------------------------------


def test_all() -> list[ConnectionResult]:
    """
    按当前配置测试所有已配置的服务。

    只测已经填了 Key 的服务：把没配置的供应商一律报成失败，会让真正的错误
    淹没在一片红色里。
    """
    results: list[ConnectionResult] = [test_llm()]

    if config.app.get("gemini_api_key") or config.app.get("gemini_image_api_key"):
        results.append(test_still_provider())
    if config.app.get("kling_api_key"):
        results.append(test_motion_provider())
    if config.app.get("sonilo_api_key"):
        results.append(test_music_provider("sonilo"))
    if config.elevenlabs.get("api_key") or config.app.get("elevenlabs_api_key"):
        results.append(test_music_provider("elevenlabs"))
    for provider in _STOCK_PROBES:
        if config.app.get(f"{provider}_api_keys"):
            results.append(test_stock_provider(provider))

    return results
