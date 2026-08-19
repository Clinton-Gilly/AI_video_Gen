"""
连载系列的落盘与续集衔接。

"Part 1 / Part 2" 的连载格式要求两件事跨进程存活：常驻角色的定妆图，以及
上一集留下的悬念。只要有一样丢了，续集就会换脸或者接不上前情。因此系列
设定、角色参考图和每一集剧本都以文件形式持久化在 ``storage/series/`` 下，
而不是留在任务状态里——任务状态会随缓存清理和进程重启消失。

目录结构::

    storage/series/<series_id>/
        series.json              系列设定与角色表
        cast/<character_id>.png  角色定妆图
        episodes/part-001.json   分集剧本

写入统一走 ``utils.write_json_atomic``，避免进程中断留下半个 JSON 让整个
系列无法加载。
"""

from __future__ import annotations

import json
import os
import shutil
import threading
from pathlib import Path
from typing import List, Optional

from loguru import logger

from app.models.story import Character, Episode, Series
from app.services import story
from app.utils import utils


# 允许的图片扩展名。定妆图由画面 provider 产出，这里只做落盘和路径校验。
_REFERENCE_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}

# 追加一集需要"读取集数 → 写入剧本 → 回写集数"三步。进程内用锁串行化，
# 避免 WebUI 与 API 并发生成同一系列时把两集写成同一个 part number。
# 多进程部署仍需外部协调，与现有任务状态的处理方式保持一致。
_series_lock = threading.RLock()


class SeriesStoreError(RuntimeError):
    """系列数据读写失败。"""


def _validate_identifier(value: str, field_name: str) -> str:
    """
    校验会参与路径拼接的标识符。

    ``Series`` 和 ``Character`` 的模型校验器已经限制了取值，但加载入口接收的是
    外部传入的裸字符串，必须在拼接路径之前独立把关。
    """
    normalized = str(value or "").strip().lower()
    if not normalized:
        raise ValueError(f"{field_name} is required")
    if not normalized.replace("-", "").replace("_", "").isalnum():
        raise ValueError(
            f"{field_name} may only contain letters, digits, '-' and '_'"
        )
    return normalized


def series_root() -> Path:
    return Path(utils.storage_dir("series", create=True))


def series_dir(series_id: str, create: bool = False) -> Path:
    directory = series_root() / _validate_identifier(series_id, "series_id")
    if create:
        directory.mkdir(parents=True, exist_ok=True)
    return directory


def _series_file(series_id: str) -> Path:
    return series_dir(series_id) / "series.json"


def _cast_dir(series_id: str, create: bool = False) -> Path:
    directory = series_dir(series_id) / "cast"
    if create:
        directory.mkdir(parents=True, exist_ok=True)
    return directory


def _episodes_dir(series_id: str, create: bool = False) -> Path:
    directory = series_dir(series_id) / "episodes"
    if create:
        directory.mkdir(parents=True, exist_ok=True)
    return directory


def _episode_file(series_id: str, part_number: int) -> Path:
    if part_number < 1:
        raise ValueError("part_number must be at least 1")
    # 零填充让目录列表按集数自然排序，便于人工排查。
    return _episodes_dir(series_id) / f"part-{part_number:03d}.json"


# --------------------------------------------------------------------------
# 系列
# --------------------------------------------------------------------------


def save_series(series: Series) -> Path:
    """写入或覆盖系列设定。"""
    series_dir(series.id, create=True)
    target = _series_file(series.id)
    utils.write_json_atomic(target, series.model_dump(mode="json"))
    logger.info(f"saved series: id={series.id}, episodes={series.episode_count}")
    return target


def load_series(series_id: str) -> Series:
    """读取系列设定，不存在时抛出 ``SeriesStoreError``。"""
    target = _series_file(series_id)
    try:
        with target.open("r", encoding="utf-8") as series_file:
            payload = json.load(series_file)
    except FileNotFoundError as exc:
        raise SeriesStoreError(f"series not found: {series_id}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise SeriesStoreError(f"failed to read series {series_id}: {exc}") from exc
    return Series(**payload)


def series_exists(series_id: str) -> bool:
    return _series_file(series_id).is_file()


def list_series() -> List[str]:
    """返回已保存的系列 ID，按名称排序。"""
    root = series_root()
    if not root.is_dir():
        return []
    return sorted(
        entry.name
        for entry in root.iterdir()
        if entry.is_dir() and (entry / "series.json").is_file()
    )


# --------------------------------------------------------------------------
# 角色定妆图
# --------------------------------------------------------------------------


def register_reference_image(
    series_id: str, character_id: str, source_path: str
) -> Character:
    """
    把生成好的定妆图收进系列目录，并写回角色档案。

    参考图必须复制而不是引用原路径：画面 provider 通常把结果放在临时目录里，
    而这张图要在这个系列的整个生命周期内反复用作一致性条件。
    """
    character_id = _validate_identifier(character_id, "character_id")
    source = Path(source_path)
    if not source.is_file():
        raise SeriesStoreError(f"reference image does not exist: {source_path}")

    extension = source.suffix.lower()
    if extension not in _REFERENCE_IMAGE_EXTENSIONS:
        raise SeriesStoreError(
            f"unsupported reference image type '{extension}', "
            f"expected one of {sorted(_REFERENCE_IMAGE_EXTENSIONS)}"
        )

    with _series_lock:
        series = load_series(series_id)
        character = series.character(character_id)
        if character is None:
            raise SeriesStoreError(
                f"character '{character_id}' is not part of series '{series_id}'"
            )

        destination = _cast_dir(series_id, create=True) / f"{character_id}{extension}"
        shutil.copyfile(source, destination)
        # 落盘时记录相对路径，storage 目录整体迁移或挂载到容器里仍然有效。
        character.reference_image = os.path.relpath(destination, series_root().parent)
        save_series(series)

    logger.success(
        f"registered reference image: series={series_id}, "
        f"character={character_id}, file={destination.name}"
    )
    return character


def reference_image_path(series: Series, character_id: str) -> Optional[Path]:
    """把角色档案里的相对路径还原成可读取的绝对路径。"""
    character = series.character(character_id)
    if character is None or not character.has_reference():
        return None
    resolved = Path(utils.storage_dir()).parent / character.reference_image
    return resolved if resolved.is_file() else None


# --------------------------------------------------------------------------
# 分集
# --------------------------------------------------------------------------


def save_episode(episode: Episode) -> Path:
    """写入或覆盖一集剧本，不改变系列的集数。"""
    _episodes_dir(episode.series_id, create=True)
    target = _episode_file(episode.series_id, episode.part_number)
    utils.write_json_atomic(target, episode.model_dump(mode="json"))
    logger.info(
        f"saved episode: series={episode.series_id}, part={episode.part_number}"
    )
    return target


def load_episode(series_id: str, part_number: int) -> Episode:
    target = _episode_file(series_id, part_number)
    try:
        with target.open("r", encoding="utf-8") as episode_file:
            payload = json.load(episode_file)
    except FileNotFoundError as exc:
        raise SeriesStoreError(
            f"episode not found: series={series_id}, part={part_number}"
        ) from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise SeriesStoreError(
            f"failed to read episode {series_id}/{part_number}: {exc}"
        ) from exc
    return Episode(**payload)


def list_episodes(series_id: str) -> List[int]:
    """返回已保存的集数，升序排列。"""
    directory = _episodes_dir(series_id)
    if not directory.is_dir():
        return []
    parts = []
    for entry in directory.iterdir():
        if entry.is_file() and entry.name.startswith("part-"):
            try:
                parts.append(int(entry.stem.removeprefix("part-")))
            except ValueError:
                # 目录里的其它文件不属于剧本，忽略即可。
                continue
    return sorted(parts)


def append_episode(episode: Episode) -> Series:
    """
    追加一集并推进系列集数。

    只有当这一集的编号正好接在已有集数之后时才推进计数，重写历史集不会把
    计数改小——否则重跑第 1 集会让后续集号全部错位。
    """
    with _series_lock:
        series = load_series(episode.series_id)
        save_episode(episode)
        if episode.part_number > series.episode_count:
            series.episode_count = episode.part_number
            save_series(series)
        return series


def latest_episode(series_id: str) -> Optional[Episode]:
    parts = list_episodes(series_id)
    if not parts:
        return None
    return load_episode(series_id, parts[-1])


def previous_cliffhanger(series_id: str, part_number: int) -> str:
    """
    取上一集留下的悬念，作为续集生成的前情输入。

    第一集、上一集缺失或上一集已完结时返回空串，由调用方按开篇处理。
    """
    if part_number <= 1:
        return ""
    try:
        previous = load_episode(series_id, part_number - 1)
    except SeriesStoreError:
        logger.warning(
            f"no previous episode to continue from: series={series_id}, "
            f"part={part_number - 1}"
        )
        return ""
    return previous.cliffhanger.strip()


# --------------------------------------------------------------------------
# 高层入口
# --------------------------------------------------------------------------


def create_series(
    concept: str,
    series_id: str,
    **kwargs,
) -> Series:
    """生成并保存一个新系列。已存在同名系列时拒绝覆盖。"""
    series_id = _validate_identifier(series_id, "series_id")
    if series_exists(series_id):
        raise SeriesStoreError(f"series already exists: {series_id}")
    series = story.generate_series(concept=concept, series_id=series_id, **kwargs)
    save_series(series)
    return series


def continue_series(
    series_id: str,
    premise: str,
    is_finale: bool = False,
    target_seconds: int = 45,
    app_config=None,
) -> Episode:
    """
    生成系列的下一集，自动接上上一集的悬念。

    这里刻意不检查定妆图是否齐备：剧本阶段不消耗画面额度，缺图应当在画面
    生成入口拦截，而不是阻断写剧本。``Series.missing_references()`` 供调用方
    在渲染前自行检查。
    """
    series = load_series(series_id)
    part_number = series.next_part_number()
    episode = story.generate_episode(
        series=series,
        premise=premise,
        part_number=part_number,
        previous_cliffhanger=previous_cliffhanger(series_id, part_number),
        is_finale=is_finale,
        target_seconds=target_seconds,
        app_config=app_config,
    )
    append_episode(episode)
    return episode
