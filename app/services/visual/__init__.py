"""
画面生成层：把结构化剧本变成逐镜画面素材。

对外只暴露三个入口：

* ``get_still_provider`` / ``get_motion_provider`` —— 按配置取出后端。
* ``generate_reference_images`` —— 为缺定妆图的角色补齐参考图。
* ``generate_episode_visuals`` —— 按镜头产出静帧，并只为计划内的镜头调用
  图生视频。

后端通过名字注册，新增供应商只需实现 ``base`` 里的协议并在这里登记一行，
不需要改动剧本层或任务流水线。
"""

from __future__ import annotations

import os
from typing import Any, Callable, List

from loguru import logger

from app.config import config
from app.models.story import Episode, Series
from app.services import series as series_store
from app.services import story
from app.services.visual.base import (
    ClipResult,
    MotionProvider,
    StillProvider,
    VisualConfigurationError,
    VisualError,
    VisualGenerationError,
)
from app.services.visual.gemini_images import GeminiImageProvider
from app.services.visual.kling_video import KlingVideoProvider
from app.services.visual.planner import (
    DEFAULT_MAX_ANIMATED_SHOTS,
    ShotPlan,
    estimate_cost,
    plan_episode,
    score_shot,
)

__all__ = [
    "DEFAULT_MAX_ANIMATED_SHOTS",
    "ClipResult",
    "MotionProvider",
    "ShotPlan",
    "StillProvider",
    "VisualConfigurationError",
    "VisualError",
    "VisualGenerationError",
    "estimate_cost",
    "generate_episode_visuals",
    "generate_reference_images",
    "get_motion_provider",
    "get_still_provider",
    "plan_episode",
    "score_shot",
]


_STILL_PROVIDERS: dict[str, Callable[[Any], StillProvider]] = {
    "gemini": GeminiImageProvider,
}
_MOTION_PROVIDERS: dict[str, Callable[[Any], MotionProvider]] = {
    "kling": KlingVideoProvider,
}

DEFAULT_STILL_PROVIDER = "gemini"
DEFAULT_MOTION_PROVIDER = "kling"


def get_still_provider(name: str = "", app_config: Any = None) -> StillProvider:
    app_config = app_config if app_config is not None else config.app
    resolved = (
        str(name or app_config.get("still_provider") or DEFAULT_STILL_PROVIDER)
        .strip()
        .lower()
    )
    factory = _STILL_PROVIDERS.get(resolved)
    if factory is None:
        raise VisualConfigurationError(
            f"unknown still provider '{resolved}', "
            f"expected one of {sorted(_STILL_PROVIDERS)}"
        )
    return factory(app_config)


def get_motion_provider(name: str = "", app_config: Any = None) -> MotionProvider:
    app_config = app_config if app_config is not None else config.app
    resolved = (
        str(name or app_config.get("motion_provider") or DEFAULT_MOTION_PROVIDER)
        .strip()
        .lower()
    )
    factory = _MOTION_PROVIDERS.get(resolved)
    if factory is None:
        raise VisualConfigurationError(
            f"unknown motion provider '{resolved}', "
            f"expected one of {sorted(_MOTION_PROVIDERS)}"
        )
    return factory(app_config)


def generate_reference_images(
    series: Series,
    output_dir: str,
    still_provider: StillProvider | None = None,
    overwrite: bool = False,
) -> List[str]:
    """
    为还没有定妆图的角色生成参考图，并登记到系列目录。

    定妆图是一次性成本，一个角色画一次就能用满整个系列，因此默认跳过已有
    参考图的角色——重画会让老剧集和新剧集里的同一个角色对不上。
    """
    provider = still_provider or get_still_provider()
    if not provider.is_enabled():
        raise VisualConfigurationError(
            f"still provider '{provider.name}' is not configured"
        )

    os.makedirs(output_dir, exist_ok=True)
    generated: List[str] = []
    for character in series.cast:
        if character.has_reference() and not overwrite:
            logger.info(f"character already has a reference image: {character.id}")
            continue

        prompt = story.build_reference_image_prompt(series, character)
        target = os.path.join(output_dir, f"{character.id}.png")
        logger.info(f"generating reference image: character={character.id}")
        provider.generate_still(
            prompt=prompt,
            output_path=target,
            aspect=series.aspect.value,
        )
        series_store.register_reference_image(series.id, character.id, target)
        generated.append(character.id)

    return generated


def generate_episode_visuals(
    series: Series,
    episode: Episode,
    output_dir: str,
    still_provider: StillProvider | None = None,
    motion_provider: MotionProvider | None = None,
    max_animated_shots: int = DEFAULT_MAX_ANIMATED_SHOTS,
) -> List[ClipResult]:
    """
    产出一集的全部画面素材。

    每个镜头都出静帧，但只有计划选中的镜头会调用图生视频——这是单集成本的
    主要控制点，其余镜头的运动留给装配阶段用 Ken Burns 免费实现。

    缺定妆图时直接拒绝：没有参考图的角色会在每个镜头里换脸，先把整集画完再
    发现要重做，代价远高于在这里停下。
    """
    missing = series.missing_references()
    if missing:
        raise VisualConfigurationError(
            "these characters have no reference image yet, so they would look "
            f"different in every shot: {', '.join(missing)}"
        )

    stills = still_provider or get_still_provider()
    if not stills.is_enabled():
        raise VisualConfigurationError(
            f"still provider '{stills.name}' is not configured"
        )

    plans = plan_episode(episode, max_animated_shots=max_animated_shots)
    needs_motion = any(plan.animate for plan in plans)
    motion = motion_provider or (get_motion_provider() if needs_motion else None)
    if needs_motion and not (motion and motion.is_enabled()):
        raise VisualConfigurationError(
            "the plan animates "
            f"{sum(1 for plan in plans if plan.animate)} shots but the motion "
            "provider is not configured; set max_animated_shots to 0 to render "
            "stills only"
        )

    os.makedirs(output_dir, exist_ok=True)
    aspect = series.aspect.value
    results: List[ClipResult] = []

    for position, plan in enumerate(plans, start=1):
        shot = plan.shot
        still_path = os.path.join(output_dir, f"shot-{position:03d}.png")
        stills.generate_still(
            prompt=story.build_shot_image_prompt(series, shot),
            output_path=still_path,
            aspect=aspect,
            reference_images=_reference_paths(series, shot),
        )

        clip_path = None
        if plan.animate:
            clip_path = os.path.join(output_dir, f"shot-{position:03d}.mp4")
            motion.animate(
                still_path=still_path,
                prompt=story.build_shot_motion_prompt(shot),
                duration=shot.duration,
                output_path=clip_path,
                aspect=aspect,
            )

        results.append(
            ClipResult(
                shot_index=position,
                still_path=still_path,
                clip_path=clip_path,
                animated=plan.animate,
                duration=shot.duration,
            )
        )

    logger.success(
        f"episode visuals ready: series={series.id}, part={episode.part_number}, "
        f"stills={len(results)}, animated={sum(1 for r in results if r.animated)}"
    )
    return results


def _reference_paths(series: Series, shot) -> list[str]:
    """取出镜头内出场角色的定妆图路径，缺失的跳过。"""
    paths = []
    for character_id in shot.character_ids:
        resolved = series_store.reference_image_path(series, character_id)
        if resolved is not None:
            paths.append(str(resolved))
    return paths
