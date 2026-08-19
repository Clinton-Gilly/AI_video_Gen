"""
角色短剧的剧本生成阶段：把一句话选题变成结构化的 ``Series`` 和 ``Episode``。

与 ``llm.generate_script`` 的区别在于产物形态。旁白短视频只需要一段自由文本，
后续用关键词去素材库找空镜；角色短剧必须先确定"谁、在哪、做什么、说什么"，
画面才能逐镜生成。因此这里要求模型返回严格的 JSON，并交给 ``app.models.story``
做结构校验——镜头编号断档、连载集没有悬念、台词的说话人不在画面里，这些错误
必须在消耗画面生成额度之前就被拦下。

本模块只产出文本：剧本结构，以及给画面层使用的提示词字符串。它不调用任何
图像或视频模型，因此可以在后端选型确定之前独立开发和测试。
"""

from __future__ import annotations

import json
import re
from typing import Any, Mapping, Optional

from loguru import logger
from pydantic import ValidationError

from app.models.schema import VideoAspect
from app.models.story import (
    MAX_CAST_SIZE,
    MAX_SCENES_PER_EPISODE,
    MAX_SHOTS_PER_EPISODE,
    Character,
    Episode,
    Series,
)

# 通过模块引用而不是 ``from ... import`` 调用大模型入口，这样测试可以直接
# 替换 ``app.services.llm._generate_response``，与现有用例的打桩方式一致。
from app.services import llm


MAX_PREMISE_LENGTH = 2000
MAX_STYLE_LENGTH = 600
# 剧本 JSON 比文案长得多，但结构固定。三次重试足以覆盖模型偶发的格式错误，
# 再多重试通常是提示词或模型能力问题，继续重试只会拉长任务时间。
MAX_STORY_RETRIES = 3


class StoryGenerationError(RuntimeError):
    """剧本生成失败：模型返回无法解析，或结构校验未通过。"""


def _limit(text: Optional[str], max_length: int, field_name: str) -> str:
    value = str(text or "").strip()
    if len(value) > max_length:
        raise ValueError(f"{field_name} exceeds {max_length} characters")
    return value


def _language_instruction(language: str) -> str:
    normalized = str(language or "").strip()
    if not normalized:
        return (
            "Write all dialogue, narration and captions in the same language as "
            "the premise."
        )
    return f"Write all dialogue, narration and captions in {normalized}."


def _parse_json_object(response: str) -> dict:
    """
    解析模型返回的 JSON 对象，并对常见的包裹形式做一次兜底。

    复用 ``llm._strip_code_fence`` 处理 ```json 围栏；如果模型还在 JSON 前后
    附加了说明文字，则退回到截取最外层花括号。两次都失败才算生成失败。
    """
    text = llm._strip_code_fence(response or "")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise StoryGenerationError("model response did not contain a JSON object")
        try:
            payload = json.loads(match.group())
        except json.JSONDecodeError as exc:
            raise StoryGenerationError(f"model returned malformed JSON: {exc}") from exc

    if not isinstance(payload, dict):
        raise StoryGenerationError("model response was not a JSON object")
    return payload


def _request_json(prompt: str, app_config=None) -> dict:
    """带重试地向模型请求一个 JSON 对象。"""
    last_error: Exception | None = None
    for attempt in range(1, MAX_STORY_RETRIES + 1):
        try:
            if app_config is None:
                response = llm._generate_response(prompt=prompt)
            else:
                response = llm._generate_response(prompt=prompt, app_config=app_config)
            if not response:
                raise StoryGenerationError("model returned an empty response")
            return _parse_json_object(response)
        except (StoryGenerationError, ValueError) as exc:
            last_error = exc
            logger.warning(
                f"story generation attempt {attempt}/{MAX_STORY_RETRIES} failed: {exc}"
            )
    raise StoryGenerationError(f"failed to generate story data: {last_error}")


# --------------------------------------------------------------------------
# 系列设定（风格圣经 + 常驻角色）
# --------------------------------------------------------------------------

_SERIES_SCHEMA = """{
  "title": "series title",
  "premise": "one paragraph describing the recurring situation",
  "moral_theme": "the value the series keeps returning to",
  "visual_style": "one sentence locking rendering style, lighting and palette",
  "cast": [
    {
      "id": "lowercase-slug",
      "name": "character name",
      "head_type": "the fruit used as the head, or an empty string for humans",
      "appearance": "fixed physical description: body, outfit, colours, accessories",
      "personality": "how they behave and speak"
    }
  ]
}"""


def build_series_prompt(
    concept: str,
    cast_size: int = 3,
    language: str = "",
    character_kind: str = "fruit",
    custom_style: str = "",
) -> str:
    style_line = (
        f"Use this visual style verbatim: {custom_style}"
        if custom_style
        else (
            "Invent a visual style that is cheap to keep consistent: a single "
            "rendering technique, one lighting setup and a small fixed palette."
        )
    )
    head_line = (
        "Every character has a photorealistic human body wearing ordinary clothes, "
        f"topped with an oversized {character_kind} in place of a head."
        if character_kind and character_kind != "human"
        else "Every character is an ordinary human."
    )
    return f"""
You are a show runner for vertical short-form moral mini-dramas.

Design a recurring series based on this concept:
{concept}

{head_line}
{style_line}
{_language_instruction(language)}

Rules:
1. Create exactly {cast_size} recurring characters.
2. "appearance" is a locked description reused in every single shot for the
   whole run of the series. Make it specific and unambiguous: body type,
   exact clothing, exact colours, one memorable accessory. Never describe a
   pose, an expression or an action there.
3. "id" must be a lowercase slug using only letters, digits and hyphens.
4. Characters must be visually distinguishable at thumbnail size.
5. Return the JSON object only. No commentary, no markdown fence.

Return this exact shape:
{_SERIES_SCHEMA}
""".strip()


def generate_series(
    concept: str,
    series_id: str,
    cast_size: int = 3,
    language: str = "",
    character_kind: str = "fruit",
    custom_style: str = "",
    aspect: VideoAspect = VideoAspect.portrait,
    narrator_voice: str = "",
    app_config=None,
) -> Series:
    """生成一个系列的设定与常驻角色表。"""
    concept = _limit(concept, MAX_PREMISE_LENGTH, "concept")
    if not concept:
        raise ValueError("concept is required")
    custom_style = _limit(custom_style, MAX_STYLE_LENGTH, "custom_style")
    if not 1 <= cast_size <= MAX_CAST_SIZE:
        raise ValueError(f"cast_size must be between 1 and {MAX_CAST_SIZE}")

    logger.info(f"generating series bible: id={series_id}, cast_size={cast_size}")
    payload = _request_json(
        build_series_prompt(
            concept=concept,
            cast_size=cast_size,
            language=language,
            character_kind=character_kind,
            custom_style=custom_style,
        ),
        app_config=app_config,
    )

    try:
        series = Series(
            id=series_id,
            title=str(payload.get("title") or concept)[:200],
            premise=str(payload.get("premise") or "")[:1000],
            moral_theme=str(payload.get("moral_theme") or "")[:300],
            visual_style=str(
                custom_style or payload.get("visual_style") or ""
            )[:MAX_STYLE_LENGTH],
            cast=[_build_character(item) for item in payload.get("cast") or []],
            aspect=aspect,
            language=language,
            narrator_voice=narrator_voice,
        )
    except ValidationError as exc:
        raise StoryGenerationError(f"generated series failed validation: {exc}") from exc

    if not series.cast:
        raise StoryGenerationError("generated series has no characters")
    logger.success(
        f"series ready: id={series.id}, cast={[m.id for m in series.cast]}"
    )
    return series


def _build_character(item: Any) -> Character:
    if not isinstance(item, Mapping):
        raise StoryGenerationError("cast entries must be JSON objects")
    return Character(
        id=str(item.get("id") or "").strip(),
        name=str(item.get("name") or "").strip(),
        head_type=str(item.get("head_type") or "").strip()[:64],
        appearance=str(item.get("appearance") or "").strip()[:600],
        personality=str(item.get("personality") or "").strip()[:300],
    )


# --------------------------------------------------------------------------
# 分集剧本
# --------------------------------------------------------------------------

_EPISODE_SCHEMA = """{
  "title": "episode title",
  "hook": "the line shown in the first three seconds",
  "moral": "the lesson this part lands on",
  "cliffhanger": "the unanswered question that forces the next part",
  "scenes": [
    {
      "beat": "hook | setup | inciting_incident | escalation | turn | consequence | resolution | cliffhanger",
      "setting": "where and when",
      "mood": "emotional temperature",
      "shots": [
        {
          "action": "what is visible in this frame, no character descriptions",
          "character_ids": ["id"],
          "shot_size": "establishing | wide | medium | close_up | extreme_close_up | over_the_shoulder | insert",
          "camera_move": "static | slow_push_in | slow_pull_out | pan_left | pan_right | handheld",
          "duration": 3.0,
          "dialogue": {"speaker_id": "id", "line": "spoken words"},
          "narration": "narrator text, empty when there is dialogue",
          "caption": "short on-screen emphasis text",
          "caption_style": "narration | dialogue | emphasis | title_card"
        }
      ]
    }
  ]
}"""


def build_episode_prompt(
    series: Series,
    premise: str,
    part_number: int,
    previous_cliffhanger: str = "",
    is_finale: bool = False,
    target_seconds: int = 45,
) -> str:
    cast_lines = "\n".join(
        f"- {member.id}: {member.name}"
        + (f" ({member.personality})" if member.personality else "")
        for member in series.cast
    )
    continuity = (
        f"""
This is Part {part_number}. Part {part_number - 1} ended on this unresolved
question, and the first scene must pay it off before moving on:
{previous_cliffhanger}
""".strip()
        if part_number > 1 and previous_cliffhanger
        else f"This is Part {part_number}, the opening part. Start cold, mid-conflict."
    )
    ending = (
        "This is the final part. Resolve the conflict, land the moral, and set "
        '"cliffhanger" to an empty string.'
        if is_finale
        else (
            "This is not the final part. It must end mid-crisis on a concrete "
            "unanswered question written into \"cliffhanger\"."
        )
    )
    return f"""
You are writing a vertical short-form moral mini-drama.

Series: {series.title}
Premise: {series.premise}
Recurring theme: {series.moral_theme}

Cast (use these ids exactly, never invent new ones):
{cast_lines}

Episode premise:
{premise}

{continuity}
{ending}
{_language_instruction(series.language)}

Rules:
1. Target about {target_seconds} seconds. Shot durations must sum to roughly
   that, with each shot between 1 and 12 seconds.
2. At most {MAX_SCENES_PER_EPISODE} scenes and {MAX_SHOTS_PER_EPISODE} shots in total.
3. Every shot needs at least one of: dialogue, narration, or caption.
4. A shot's "dialogue.speaker_id" must also appear in that shot's "character_ids".
5. "action" describes only what happens in the frame. Never describe what a
   character looks like: their appearance is fixed elsewhere and repeating it
   causes them to drift between shots.
6. Open on conflict already in progress. No introductions, no throat-clearing.
7. Return the JSON object only. No commentary, no markdown fence.

Return this exact shape:
{_EPISODE_SCHEMA}
""".strip()


def generate_episode(
    series: Series,
    premise: str,
    part_number: int = 1,
    previous_cliffhanger: str = "",
    is_finale: bool = False,
    target_seconds: int = 45,
    app_config=None,
) -> Episode:
    """生成一集的结构化剧本，并按系列角色表校验说话人。"""
    premise = _limit(premise, MAX_PREMISE_LENGTH, "premise")
    if not premise:
        raise ValueError("premise is required")
    if part_number < 1:
        raise ValueError("part_number must be at least 1")
    if not series.cast:
        raise ValueError("series has no cast")

    logger.info(
        f"generating episode: series={series.id}, part={part_number}, "
        f"is_finale={is_finale}, target_seconds={target_seconds}"
    )
    payload = _request_json(
        build_episode_prompt(
            series=series,
            premise=premise,
            part_number=part_number,
            previous_cliffhanger=previous_cliffhanger,
            is_finale=is_finale,
            target_seconds=target_seconds,
        ),
        app_config=app_config,
    )

    scenes = payload.get("scenes")
    if not isinstance(scenes, list) or not scenes:
        raise StoryGenerationError("generated episode contains no scenes")

    normalized_scenes = [
        _normalize_scene(scene, index) for index, scene in enumerate(scenes, start=1)
    ]
    try:
        episode = Episode(
            series_id=series.id,
            part_number=part_number,
            title=str(payload.get("title") or premise)[:200],
            premise=premise,
            hook=str(payload.get("hook") or "")[:200],
            moral=str(payload.get("moral") or "")[:300],
            scenes=normalized_scenes,
            cliffhanger=str(payload.get("cliffhanger") or "")[:300],
            is_finale=is_finale,
        )
    except ValidationError as exc:
        raise StoryGenerationError(
            f"generated episode failed validation: {exc}"
        ) from exc

    _validate_against_cast(series, episode)
    logger.success(
        f"episode ready: series={series.id}, part={part_number}, "
        f"shots={len(episode.shots())}, duration={episode.duration():.1f}s"
    )
    return episode


def _normalize_scene(scene: Any, index: int) -> dict:
    """补齐模型常漏的字段，并把镜头重新编号成连续序列。"""
    if not isinstance(scene, Mapping):
        raise StoryGenerationError(f"scene {index} is not a JSON object")
    shots = scene.get("shots")
    if not isinstance(shots, list) or not shots:
        raise StoryGenerationError(f"scene {index} contains no shots")

    normalized_shots = []
    for shot_index, shot in enumerate(shots, start=1):
        if not isinstance(shot, Mapping):
            raise StoryGenerationError(
                f"scene {index} shot {shot_index} is not a JSON object"
            )
        # 模型常把 index 写错或漏写。镜头顺序由数组顺序决定，直接重编号比
        # 让结构校验失败后整集重来更划算。
        normalized = dict(shot)
        normalized["index"] = shot_index
        dialogue = normalized.get("dialogue")
        # 无台词镜头经常返回 null、空串或空对象，统一归一成缺省值。
        if not isinstance(dialogue, Mapping) or not str(
            dialogue.get("line") or ""
        ).strip():
            normalized["dialogue"] = None
        normalized_shots.append(normalized)

    return {
        "index": index,
        "beat": scene.get("beat") or "escalation",
        "setting": str(scene.get("setting") or "").strip() or "unspecified location",
        "mood": str(scene.get("mood") or "")[:120],
        "shots": normalized_shots,
    }


def _validate_against_cast(series: Series, episode: Episode) -> None:
    """
    拦下引用了不存在角色的剧本。

    未知角色 ID 意味着画面层拿不到对应的定妆图，只能退化成纯文本提示词，
    该角色会在每个镜头里长得都不一样——这正是这套模型要防的失败模式。
    """
    known = {member.id for member in series.cast}
    unknown = sorted(set(episode.character_ids()) - known)
    if unknown:
        raise StoryGenerationError(
            f"episode references characters that are not in the series cast: "
            f"{', '.join(unknown)}"
        )


# --------------------------------------------------------------------------
# 画面提示词（供后续的图生视频 provider 消费）
# --------------------------------------------------------------------------


def build_reference_image_prompt(series: Series, character: Character) -> str:
    """
    生成角色定妆图的提示词。

    定妆图是整条一致性链路的锚点：先按锁定描述生成一张中性站姿的角色图，
    之后每个镜头的静帧都以它为条件生成，角色才不会逐镜漂移。因此这里刻意
    要求中性姿势、平光和纯色背景——参考图里的姿势、光线和背景都会被后续
    生成继承，越中性可复用性越强。
    """
    style = series.visual_style or "consistent cinematic rendering"
    return (
        f"Character reference sheet of {character.to_prompt()}. "
        f"Neutral standing pose, arms at sides, facing the camera, neutral "
        f"expression. Even flat lighting, plain seamless mid-grey background. "
        f"Full body in frame. {style}. "
        f"No text, no watermark, no logo, no border, no collage."
    )


def build_shot_image_prompt(series: Series, shot) -> str:
    """
    组装单个镜头的静帧提示词。

    角色长相不取自镜头本身，而是按 ID 从系列角色表回填，保证同一角色在每个
    镜头里得到逐字一致的描述；风格圣经同样逐字附加。镜头只贡献动作、镜别和
    场景信息。
    """
    described = []
    for character_id in shot.character_ids:
        member = series.character(character_id)
        if member is None:
            # 剧本已在 ``_validate_against_cast`` 校验过；这里是给直接调用
            # 本函数的调用方（例如手工拼装的镜头）留的防线。
            raise ValueError(f"unknown character id: {character_id}")
        described.append(member.to_prompt())

    parts = [shot.shot_size.to_prompt()]
    if described:
        parts.append("featuring " + "; ".join(described))
    parts.append(shot.action)
    if series.visual_style:
        parts.append(series.visual_style)
    parts.append("no text, no subtitles, no watermark, no logo")
    return ". ".join(part.strip().rstrip(".") for part in parts if part.strip()) + "."


def build_shot_motion_prompt(shot) -> str:
    """
    组装图生视频阶段的运动提示词。

    只描述运动，不复述画面内容：画面已经由输入的静帧决定，再复述一遍会诱使
    视频模型重新构图，反而破坏与静帧的一致性。
    """
    parts = [shot.camera_move.to_prompt()]
    if shot.dialogue:
        parts.append("the character is speaking, natural lip and head movement")
    else:
        parts.append("subtle ambient motion only")
    return ". ".join(parts) + "."
