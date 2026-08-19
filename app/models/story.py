"""
角色化短剧（fruit-character mini-drama）的结构化数据模型。

现有 ``VideoParams`` 面向"旁白 + 素材库空镜"流程：文案是一整段字符串，
镜头顺序可以随机打乱。角色短剧的约束完全不同——同一批角色要在几十个镜头
里保持长相一致，镜头必须按叙事顺序播放，台词要按角色分配不同音色，并且
Part 1 的悬念要能被 Part 2 接上。

因此这里引入独立的 ``Series → Episode → Scene → Shot`` 模型：

* ``Series`` 持有跨集不变的部分：视觉风格圣经、常驻角色表、画幅、语言。
* ``Episode`` 是一次投稿，携带钩子、道德落点和给下一集的悬念。
* ``Scene`` 是一个叙事节拍，锁定场景和情绪。
* ``Shot`` 是一次画面生成的最小单位，直接对应"生成静帧 → 驱动成片"。

这些模型只描述剧本结构，不负责生成图像或视频，也不写盘：落盘由
``app.services.series`` 负责，画面生成由后续的 provider 层负责。
"""

from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

from app.models.schema import VideoAspect


# 单集镜头总数的硬上限。角色短剧按 2-4 秒一个镜头计算，60 个镜头已经覆盖
# 约三分钟的成片；再长的输入通常是模型跑飞，继续生成只会浪费画面额度。
MAX_SHOTS_PER_EPISODE = 60
MAX_SCENES_PER_EPISODE = 12
MAX_CAST_SIZE = 8

# 单个镜头的时长边界。短于 1 秒的镜头在竖屏信息流里来不及被看清，
# 长于 12 秒则超出主流图生视频模型的单次生成上限。
MIN_SHOT_SECONDS = 1.0
MAX_SHOT_SECONDS = 12.0


class ShotSize(str, Enum):
    """镜别。生成静帧时直接翻译成构图提示词。"""

    establishing = "establishing"
    wide = "wide"
    medium = "medium"
    close_up = "close_up"
    extreme_close_up = "extreme_close_up"
    over_the_shoulder = "over_the_shoulder"
    insert = "insert"

    def to_prompt(self) -> str:
        return _SHOT_SIZE_PROMPTS[self]


_SHOT_SIZE_PROMPTS = {
    ShotSize.establishing: "wide establishing shot showing the full location",
    ShotSize.wide: "wide shot, full body visible",
    ShotSize.medium: "medium shot from the waist up",
    ShotSize.close_up: "close-up on the face, shallow depth of field",
    ShotSize.extreme_close_up: "extreme close-up on the eyes",
    ShotSize.over_the_shoulder: "over-the-shoulder shot framing the other character",
    ShotSize.insert: "insert shot of a significant object, no faces",
}


class CameraMove(str, Enum):
    """镜头运动。图生视频阶段作为运动提示词，静帧回退时映射为 Ken Burns。"""

    static = "static"
    slow_push_in = "slow_push_in"
    slow_pull_out = "slow_pull_out"
    pan_left = "pan_left"
    pan_right = "pan_right"
    handheld = "handheld"

    def to_prompt(self) -> str:
        return _CAMERA_MOVE_PROMPTS[self]


_CAMERA_MOVE_PROMPTS = {
    CameraMove.static: "locked-off static camera",
    CameraMove.slow_push_in: "slow dolly push in",
    CameraMove.slow_pull_out: "slow dolly pull out",
    CameraMove.pan_left: "slow pan to the left",
    CameraMove.pan_right: "slow pan to the right",
    CameraMove.handheld: "subtle handheld camera shake",
}


class StoryBeat(str, Enum):
    """
    叙事节拍。道德短剧的爆款结构高度模板化，把节拍显式建模有三个作用：
    校验大模型是否真的写出了完整弧线、决定悬念该切在哪里、以及让配乐和
    字幕强调按节拍变化而不是逐句变化。
    """

    hook = "hook"
    setup = "setup"
    inciting_incident = "inciting_incident"
    escalation = "escalation"
    turn = "turn"
    consequence = "consequence"
    resolution = "resolution"
    cliffhanger = "cliffhanger"


class CaptionStyle(str, Enum):
    """字幕呈现方式。旁白字幕和短剧的强调字幕是两种不同的视觉语言。"""

    narration = "narration"
    dialogue = "dialogue"
    emphasis = "emphasis"
    title_card = "title_card"


class Character(BaseModel):
    """
    常驻角色档案。

    ``appearance`` 一旦确定就不应再改写：它是每个镜头提示词都会原样复述的
    锁定描述，也是参考图的生成依据。``reference_image`` 指向该角色的定妆图，
    图生视频阶段用它做一致性条件——纯文本提示词无法让同一个草莓头角色在
    几十个镜头里保持同一张脸。
    """

    id: str = Field(min_length=1, max_length=64)
    name: str = Field(min_length=1, max_length=64)
    # 头部造型，例如 "strawberry"、"dragonfruit"；写实真人系列留空即可。
    head_type: str = Field(default="", max_length=64)
    appearance: str = Field(default="", max_length=600)
    personality: str = Field(default="", max_length=300)
    # 角色专属音色，留空时回退到剧集的旁白音色。
    voice_name: str = Field(default="", max_length=128)
    reference_image: str = Field(default="", max_length=1024)

    @field_validator("id")
    @classmethod
    def _validate_id(cls, value: str) -> str:
        # 角色 ID 会直接参与拼接落盘路径，因此只允许小写字母、数字、连字符
        # 和下划线，从源头上排除 `..`、路径分隔符和空白造成的目录逃逸。
        normalized = value.strip().lower()
        if not normalized.replace("-", "").replace("_", "").isalnum():
            raise ValueError(
                "character id may only contain letters, digits, '-' and '_'"
            )
        return normalized

    def has_reference(self) -> bool:
        return bool(self.reference_image.strip())

    def to_prompt(self) -> str:
        """把角色档案压成一段可以嵌进画面提示词的定型描述。"""
        parts = [self.name]
        if self.head_type:
            parts.append(f"a character with a {self.head_type} for a head")
        if self.appearance:
            parts.append(self.appearance)
        return ", ".join(parts)


class Dialogue(BaseModel):
    """一句台词。``speaker_id`` 决定用哪个角色的音色配音。"""

    speaker_id: str = Field(min_length=1, max_length=64)
    line: str = Field(min_length=1, max_length=400)


class Shot(BaseModel):
    """
    一次画面生成的最小单位。

    ``action`` 只描述这一格画面里看得见的内容，不含角色长相——长相由
    ``character_ids`` 指向的档案在组装提示词时统一注入，避免同一个角色在不同
    镜头里被大模型改写外形。
    """

    index: int = Field(ge=1)
    action: str = Field(min_length=1, max_length=600)
    character_ids: List[str] = Field(default_factory=list)
    shot_size: ShotSize = ShotSize.medium
    camera_move: CameraMove = CameraMove.static
    duration: float = Field(default=3.0, ge=MIN_SHOT_SECONDS, le=MAX_SHOT_SECONDS)
    dialogue: Optional[Dialogue] = None
    narration: str = Field(default="", max_length=400)
    # 压在画面上的强调字幕，通常是一到四个词的钩子文本。
    caption: str = Field(default="", max_length=120)
    caption_style: CaptionStyle = CaptionStyle.narration

    @model_validator(mode="after")
    def _require_spoken_or_visual_content(self) -> "Shot":
        # 既没有台词、旁白也没有强调字幕的镜头会在成片里变成一段静默留白。
        # 这是大模型漏填字段的典型症状，放行到渲染阶段才发现代价更高。
        if not (self.dialogue or self.narration.strip() or self.caption.strip()):
            raise ValueError(
                f"shot {self.index} has no dialogue, narration or caption"
            )
        if self.dialogue and self.dialogue.speaker_id not in self.character_ids:
            raise ValueError(
                f"shot {self.index} speaker '{self.dialogue.speaker_id}' "
                "is not present in the shot"
            )
        return self

    def spoken_text(self) -> str:
        """返回该镜头需要配音的文本，台词优先于旁白。"""
        if self.dialogue:
            return self.dialogue.line
        return self.narration.strip()


class Scene(BaseModel):
    """一个叙事节拍，锁定发生地点和情绪。"""

    index: int = Field(ge=1)
    beat: StoryBeat
    setting: str = Field(min_length=1, max_length=300)
    mood: str = Field(default="", max_length=120)
    shots: List[Shot] = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_shot_indexes(self) -> "Scene":
        expected = list(range(1, len(self.shots) + 1))
        if [shot.index for shot in self.shots] != expected:
            raise ValueError(
                f"scene {self.index} shots must be numbered 1..{len(self.shots)} in order"
            )
        return self

    def duration(self) -> float:
        return sum(shot.duration for shot in self.shots)


class Episode(BaseModel):
    """
    一集，也就是一次投稿的内容。

    ``part_number`` 和 ``cliffhanger`` 共同支撑 "Part 1 / Part 2" 结构：
    非完结集必须留下悬念，续集则在生成时把上一集的悬念作为前情输入。
    """

    series_id: str = Field(min_length=1, max_length=64)
    part_number: int = Field(default=1, ge=1)
    title: str = Field(min_length=1, max_length=200)
    premise: str = Field(default="", max_length=1000)
    # 开场三秒的钩子文本，决定信息流里的完播率。
    hook: str = Field(default="", max_length=200)
    moral: str = Field(default="", max_length=300)
    scenes: List[Scene] = Field(min_length=1, max_length=MAX_SCENES_PER_EPISODE)
    cliffhanger: str = Field(default="", max_length=300)
    is_finale: bool = False

    @model_validator(mode="after")
    def _validate_structure(self) -> "Episode":
        expected = list(range(1, len(self.scenes) + 1))
        if [scene.index for scene in self.scenes] != expected:
            raise ValueError(
                f"scenes must be numbered 1..{len(self.scenes)} in order"
            )

        shot_count = sum(len(scene.shots) for scene in self.scenes)
        if shot_count > MAX_SHOTS_PER_EPISODE:
            raise ValueError(
                f"episode has {shot_count} shots, the limit is {MAX_SHOTS_PER_EPISODE}"
            )

        # 连载集缺少悬念会让观众没有理由等下一集，这正是该格式的核心机制，
        # 因此当成结构错误拦下，而不是留给人工检查。
        if not self.is_finale and not self.cliffhanger.strip():
            raise ValueError("a non-finale episode must end on a cliffhanger")
        return self

    def shots(self) -> List[Shot]:
        """按叙事顺序展开全部镜头。渲染层据此逐镜生成画面。"""
        return [shot for scene in self.scenes for shot in scene.shots]

    def duration(self) -> float:
        return sum(scene.duration() for scene in self.scenes)

    def character_ids(self) -> List[str]:
        """按首次出场顺序返回出现过的角色 ID。"""
        ordered: List[str] = []
        for shot in self.shots():
            for character_id in shot.character_ids:
                if character_id not in ordered:
                    ordered.append(character_id)
        return ordered


class Series(BaseModel):
    """
    一个连载系列，持有跨集不变的设定。

    ``visual_style`` 是风格圣经：它会被逐字拼进每一个镜头的画面提示词，
    让不同集、不同镜头的成片看起来出自同一个频道。
    """

    id: str = Field(min_length=1, max_length=64)
    title: str = Field(min_length=1, max_length=200)
    premise: str = Field(default="", max_length=1000)
    moral_theme: str = Field(default="", max_length=300)
    visual_style: str = Field(default="", max_length=600)
    cast: List[Character] = Field(default_factory=list, max_length=MAX_CAST_SIZE)
    aspect: VideoAspect = VideoAspect.portrait
    language: str = Field(default="", max_length=32)
    # 无台词镜头和默认旁白使用的音色。
    narrator_voice: str = Field(default="", max_length=128)
    episode_count: int = Field(default=0, ge=0)

    @field_validator("id")
    @classmethod
    def _validate_id(cls, value: str) -> str:
        # 与角色 ID 同理：系列 ID 也会参与落盘路径拼接。
        normalized = value.strip().lower()
        if not normalized.replace("-", "").replace("_", "").isalnum():
            raise ValueError("series id may only contain letters, digits, '-' and '_'")
        return normalized

    @model_validator(mode="after")
    def _validate_cast(self) -> "Series":
        ids = [character.id for character in self.cast]
        if len(ids) != len(set(ids)):
            raise ValueError("series cast contains duplicate character ids")
        return self

    def character(self, character_id: str) -> Optional[Character]:
        for member in self.cast:
            if member.id == character_id:
                return member
        return None

    def next_part_number(self) -> int:
        return self.episode_count + 1

    def missing_references(self) -> List[str]:
        """返回还没有定妆图的角色 ID。缺参考图就无法保证跨镜头一致性。"""
        return [member.id for member in self.cast if not member.has_reference()]
