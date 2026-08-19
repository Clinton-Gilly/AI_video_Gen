"""
Gemini 静帧生成。

选它做静帧有两个现实理由：``google-genai`` 已经是项目依赖（LLM 层在用），
接入不需要引入新的 SDK；以及它支持把参考图和文字提示一起送进请求，这正是
角色跨镜头保持一致所依赖的条件生成能力。

本模块只负责"按提示词出一张图"，不关心提示词怎么拼——提示词由
``app.services.story`` 用锁定的角色档案和风格圣经组装。
"""

from __future__ import annotations

import mimetypes
from pathlib import Path
from typing import Any, Sequence

from loguru import logger

from app.config import config
from app.services.visual.base import (
    MAX_STILL_BYTES,
    StillResult,
    VisualConfigurationError,
    VisualGenerationError,
    validate_aspect,
    write_bytes,
)


DEFAULT_MODEL = "gemini-3-pro-image-preview"
# 参考图张数上限。角色短剧一个镜头最多同框两三个角色，再多既超出模型的
# 有效条件范围，也会显著拉高单张成本。
MAX_REFERENCE_IMAGES = 4
_SUPPORTED_REFERENCE_TYPES = {".png", ".jpg", ".jpeg", ".webp"}


class GeminiImageProvider:
    """用 Gemini 生成静帧，可选地以角色定妆图做一致性条件。"""

    name = "gemini"

    def __init__(self, app_config: Any = None):
        # 与 LLM 层一致：允许调用方传入配置快照，避免长任务运行期间用户改了
        # 配置导致同一集的前后镜头用上不同模型。
        self._config = app_config if app_config is not None else config.app

    def _api_key(self) -> str:
        # 复用 LLM 层已有的 Gemini Key，避免用户为同一个账号配置两次。
        key = str(
            self._config.get("gemini_image_api_key")
            or self._config.get("gemini_api_key")
            or ""
        ).strip()
        return key

    def _model(self) -> str:
        return (
            str(self._config.get("gemini_image_model") or "").strip() or DEFAULT_MODEL
        )

    def is_enabled(self) -> bool:
        return bool(self._api_key())

    def generate_still(
        self,
        *,
        prompt: str,
        output_path: str,
        aspect: str = "9:16",
        reference_images: Sequence[str] = (),
    ) -> StillResult:
        if not str(prompt or "").strip():
            raise VisualGenerationError("prompt is required")
        api_key = self._api_key()
        if not api_key:
            raise VisualConfigurationError(
                "Gemini image generation requires gemini_api_key in config.toml"
            )
        aspect = validate_aspect(aspect)

        # SDK 延迟导入，让没有配置画面后端的部署不必安装该依赖。
        from google import genai
        from google.genai import types

        contents: list[Any] = []
        for reference in list(reference_images)[:MAX_REFERENCE_IMAGES]:
            contents.append(self._load_reference(reference))
        # 参考图在前、文字在后：模型据此把前面的图片理解为"要保持的样子"，
        # 后面的文字理解为"这一镜要发生什么"。
        contents.append(prompt)

        model = self._model()
        logger.info(
            f"generating still: model={model}, aspect={aspect}, "
            f"references={len(contents) - 1}"
        )
        try:
            with genai.Client(api_key=api_key) as client:
                response = client.models.generate_content(
                    model=model,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        response_modalities=["IMAGE"],
                        image_config=types.ImageConfig(aspect_ratio=aspect),
                    ),
                )
        except Exception as exc:
            raise VisualGenerationError(
                f"Gemini image generation failed: {type(exc).__name__}: {exc}"
            ) from exc

        payload = _extract_image_bytes(response)
        path = write_bytes(output_path, payload, MAX_STILL_BYTES)
        return StillResult(path=path, provider=self.name, model=model)

    def _load_reference(self, reference_path: str):
        """把定妆图读成 SDK 可以直接消费的内容块。"""
        from google.genai import types

        path = Path(reference_path)
        if not path.is_file():
            raise VisualGenerationError(f"reference image not found: {reference_path}")
        suffix = path.suffix.lower()
        if suffix not in _SUPPORTED_REFERENCE_TYPES:
            raise VisualGenerationError(
                f"unsupported reference image type '{suffix}', "
                f"expected one of {sorted(_SUPPORTED_REFERENCE_TYPES)}"
            )
        mime_type = mimetypes.types_map.get(suffix) or "image/png"
        return types.Part.from_bytes(data=path.read_bytes(), mime_type=mime_type)


def _extract_image_bytes(response: Any) -> bytes:
    """
    从 SDK 响应里取出图片数据。

    模型可能在图片之外附带解释文本，也可能因为安全策略只返回文本。这里遍历
    所有分块找第一份 inline 数据，找不到时把返回的文字一并报出来，否则用户
    只会看到一句无从排查的"生成失败"。
    """
    candidates = getattr(response, "candidates", None) or []
    text_parts: list[str] = []
    for candidate in candidates:
        content = getattr(candidate, "content", None)
        for part in getattr(content, "parts", None) or []:
            inline = getattr(part, "inline_data", None)
            data = getattr(inline, "data", None)
            if data:
                return data
            text = getattr(part, "text", None)
            if text:
                text_parts.append(str(text))

    detail = " ".join(text_parts).strip()
    raise VisualGenerationError(
        "Gemini returned no image"
        + (f": {detail[:300]}" if detail else " and no explanation")
    )
