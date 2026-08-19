"""
画面生成 provider 的公共契约。

角色短剧的画面分两步产出：先按镜头生成静帧，再把需要动起来的静帧驱动成
短视频。这两步由不同厂商提供，成本量级也相差一个数量级，因此拆成两个独立
协议，可以分别替换、分别关闭。

所有 provider 都遵循同一组约定：

* ``is_enabled()`` 只检查配置是否齐备，不发起网络请求，供上层做前置校验。
* 生成方法把产物写到调用方指定的路径并返回该路径，不自行决定存储位置。
* 失败一律抛出 ``VisualError`` 的子类，避免上层需要区分 requests、SDK 和
  协议错误。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence, runtime_checkable

import requests
from loguru import logger


# 单个产物的体积上限。静帧和 3-5 秒的短片都远小于该值；超出通常意味着拿到了
# 错误的链接（例如整集成片或一个 HTML 错误页），继续写盘只会浪费磁盘。
MAX_STILL_BYTES = 20 * 1024 * 1024
MAX_CLIP_BYTES = 200 * 1024 * 1024

SUPPORTED_ASPECTS = frozenset({"9:16", "16:9", "1:1"})


class VisualError(RuntimeError):
    """画面生成失败的基类。"""


class VisualConfigurationError(VisualError):
    """provider 未配置或配置不完整。"""


class VisualGenerationError(VisualError):
    """provider 已配置，但本次生成失败。"""


@dataclass(frozen=True)
class StillResult:
    """一张生成好的静帧。"""

    path: str
    provider: str
    model: str


@dataclass(frozen=True)
class ClipResult:
    """
    一个镜头的画面产物。

    ``animated`` 区分两种来源：``True`` 表示调用了图生视频模型，``False``
    表示只有静帧，运动交给装配阶段用 Ken Burns 免费实现。成本几乎全部来自
    前者，因此这个标记也是核算单集成本的依据。
    """

    shot_index: int
    still_path: str
    clip_path: str | None
    animated: bool
    duration: float


@runtime_checkable
class StillProvider(Protocol):
    """按提示词生成静帧，可选地以参考图做一致性条件。"""

    name: str

    def is_enabled(self) -> bool: ...

    def generate_still(
        self,
        *,
        prompt: str,
        output_path: str,
        aspect: str = "9:16",
        reference_images: Sequence[str] = (),
    ) -> StillResult: ...


@runtime_checkable
class MotionProvider(Protocol):
    """把一张静帧驱动成一段短视频。"""

    name: str

    def is_enabled(self) -> bool: ...

    def animate(
        self,
        *,
        still_path: str,
        prompt: str,
        duration: float,
        output_path: str,
        aspect: str = "9:16",
    ) -> str: ...


def validate_aspect(aspect: str) -> str:
    normalized = str(aspect or "").strip()
    if normalized not in SUPPORTED_ASPECTS:
        raise VisualGenerationError(
            f"unsupported aspect ratio '{aspect}', "
            f"expected one of {sorted(SUPPORTED_ASPECTS)}"
        )
    return normalized


def ensure_parent(output_path: str) -> Path:
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def write_bytes(output_path: str, payload: bytes, max_bytes: int) -> str:
    """
    把生成结果写盘，并在写入前校验体积。

    先写临时文件再改名，避免生成中断后留下一个体积不完整的产物被后续步骤
    当成有效素材使用。
    """
    if not payload:
        raise VisualGenerationError("provider returned an empty payload")
    if len(payload) > max_bytes:
        raise VisualGenerationError(
            f"generated asset is {len(payload)} bytes, over the {max_bytes} limit"
        )

    target = ensure_parent(output_path)
    temp_path = target.with_name(f".{target.name}.tmp")
    try:
        temp_path.write_bytes(payload)
        os.replace(temp_path, target)
    finally:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)
    return str(target)


def download_to_file(
    url: str,
    output_path: str,
    max_bytes: int,
    timeout: tuple[int, int] = (10, 300),
) -> str:
    """
    流式下载 provider 产物，边下边计体积。

    先读完整个响应再判断大小会让一个异常的超大响应直接占满内存，因此按块累加
    并在超限时立刻中断。同样先写临时文件再改名。
    """
    if not str(url or "").strip():
        raise VisualGenerationError("provider returned an empty result url")

    target = ensure_parent(output_path)
    temp_path = target.with_name(f".{target.name}.tmp")
    total = 0
    try:
        with requests.get(url, stream=True, timeout=timeout) as response:
            if response.status_code != 200:
                raise VisualGenerationError(
                    f"failed to download generated asset: HTTP {response.status_code}"
                )
            with temp_path.open("wb") as output:
                for chunk in response.iter_content(chunk_size=1024 * 256):
                    if not chunk:
                        continue
                    total += len(chunk)
                    if total > max_bytes:
                        raise VisualGenerationError(
                            f"generated asset exceeds the {max_bytes} byte limit"
                        )
                    output.write(chunk)
        if total == 0:
            raise VisualGenerationError("downloaded asset is empty")
        os.replace(temp_path, target)
    except requests.RequestException as exc:
        raise VisualGenerationError(f"failed to download generated asset: {exc}") from exc
    finally:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)

    logger.info(f"downloaded generated asset: file={target.name}, bytes={total}")
    return str(target)
