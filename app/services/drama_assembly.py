"""
把逐镜画面素材装配成成片。

与素材库流程的装配有三处本质差异：

* **时长由配音决定，不由剧本决定。** 剧本里的 ``duration`` 是写稿时的估计，
  真正的镜头长度取决于这句台词念出来有多长。按剧本时长切会让台词被截断，
  因此这里一律以合成出的音频长度为准。
* **每个角色用自己的音色。** ``Character.voice_name`` 从建模起就存在，装配
  阶段是第一个真正读它的地方；没有配置的角色回退到系列旁白音色。
* **字幕是戏剧字幕，不是旁白字幕。** 强调字幕、台词和标题卡三种呈现方式
  不同，统一按一种样式压制会让成片看起来像配了旁白的幻灯片。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Sequence

from loguru import logger
from moviepy import (
    AudioFileClip,
    CompositeAudioClip,
    CompositeVideoClip,
    TextClip,
    afx,
    concatenate_videoclips,
)

from app.models.schema import VideoAspect, VideoParams
from app.models.story import CaptionStyle, Episode, Series, Shot
from app.services import bgm as bgm_service
from app.services import video as video_service
from app.services import voice as voice_service
from app.services.utils.video_effects import ken_burns_clip
from app.services.visual.base import ClipResult
from app.utils import utils


# 没有配音的镜头（只有强调字幕）仍需要一个可读的停留时间。低于这个值观众
# 来不及看清字幕，高于则拖慢节奏。
MIN_SILENT_SHOT_SECONDS = 1.2

# 字幕样式与字号的倍率关系。强调字幕和标题卡要压过画面，台词和旁白只需清晰。
_CAPTION_SCALE = {
    CaptionStyle.narration: 1.0,
    CaptionStyle.dialogue: 1.0,
    CaptionStyle.emphasis: 1.35,
    CaptionStyle.title_card: 1.6,
}

# 字幕垂直位置（相对画面高度）。标题卡居中，其余压在下方三分之一处，
# 避免遮住角色面部——竖屏短剧里面部通常在画面上半部。
_CAPTION_POSITION = {
    CaptionStyle.narration: 0.72,
    CaptionStyle.dialogue: 0.72,
    CaptionStyle.emphasis: 0.62,
    CaptionStyle.title_card: 0.45,
}


class DramaAssemblyError(RuntimeError):
    """装配失败。"""


@dataclass
class ShotAudio:
    """一个镜头的配音结果。"""

    shot_index: int
    audio_path: str | None
    duration: float
    voice_name: str


def resolve_voice(series: Series, shot: Shot, fallback: str = "") -> str:
    """
    决定这个镜头用谁的音色。

    有台词时用说话角色自己的音色，这是让多角色短剧听起来像对话而不是一个人
    念稿的关键。角色没有配置音色，或者这一镜是旁白时，回退到系列旁白音色。
    """
    if shot.dialogue:
        speaker = series.character(shot.dialogue.speaker_id)
        if speaker is not None and speaker.voice_name.strip():
            return speaker.voice_name.strip()
    return (series.narrator_voice or fallback or "").strip()


def synthesize_shot_audio(
    series: Series,
    episode: Episode,
    output_dir: str,
    params: VideoParams,
) -> List[ShotAudio]:
    """
    为每个有台词或旁白的镜头合成配音。

    只有强调字幕、没有任何念白的镜头不合成音频，交由调用方按剧本时长处理。
    单个镜头合成失败不会中断整集：该镜头退化成无声镜头，其余镜头照常产出，
    这比让一整集因为一句话失败而全部作废更实用。
    """
    os.makedirs(output_dir, exist_ok=True)
    results: List[ShotAudio] = []

    for position, shot in enumerate(episode.shots(), start=1):
        text = shot.spoken_text()
        voice_name = resolve_voice(series, shot, params.voice_name or "")
        if not text:
            results.append(
                ShotAudio(
                    shot_index=position,
                    audio_path=None,
                    duration=max(shot.duration, MIN_SILENT_SHOT_SECONDS),
                    voice_name=voice_name,
                )
            )
            continue

        audio_path = os.path.join(output_dir, f"shot-{position:03d}.mp3")
        sub_maker = voice_service.tts(
            text=text,
            voice_name=voice_name,
            voice_rate=params.voice_rate,
            voice_file=audio_path,
            voice_volume=params.voice_volume,
        )
        if sub_maker is None or not os.path.exists(audio_path):
            logger.warning(
                f"failed to synthesize shot {position}, continuing without audio"
            )
            results.append(
                ShotAudio(
                    shot_index=position,
                    audio_path=None,
                    duration=max(shot.duration, MIN_SILENT_SHOT_SECONDS),
                    voice_name=voice_name,
                )
            )
            continue

        results.append(
            ShotAudio(
                shot_index=position,
                audio_path=audio_path,
                duration=_audio_duration(audio_path, shot.duration),
                voice_name=voice_name,
            )
        )

    return results


def _audio_duration(audio_path: str, fallback: float) -> float:
    """读出音频真实时长，读取失败时退回剧本估计值。"""
    try:
        with AudioFileClip(audio_path) as clip:
            duration = float(clip.duration or 0)
        if duration > 0:
            return duration
    except Exception as exc:
        logger.warning(f"failed to read audio duration for {audio_path}: {exc}")
    return max(fallback, MIN_SILENT_SHOT_SECONDS)


def build_shot_visual(
    clip_result: ClipResult,
    shot: Shot,
    duration: float,
    resolution: tuple[int, int],
):
    """
    把一个镜头的画面素材铺满目标时长。

    动过的镜头用生成的短片：供应商按固定档位出片（通常 5 秒），比台词长就
    截断，比台词短就定格最后一帧补足——循环播放会出现明显的跳帧回跳，在
    对话镜头里尤其突兀。没动过的镜头直接用静帧走 Ken Burns，时长任意可控。
    """
    if clip_result.clip_path and os.path.exists(clip_result.clip_path):
        clip = video_service._open_video_clip_quietly(clip_result.clip_path)
        clip = clip.resized(new_size=resolution)
        if clip.duration > duration:
            return clip.subclipped(0, duration)
        if clip.duration < duration:
            # 用最后一帧定格补足剩余时间。
            frozen = clip.to_ImageClip(t=max(clip.duration - 0.05, 0)).with_duration(
                duration - clip.duration
            )
            return concatenate_videoclips([clip, frozen])
        return clip

    return ken_burns_clip(
        image_path=clip_result.still_path,
        duration=duration,
        camera_move=shot.camera_move.value,
        resolution=resolution,
    )


def build_caption_clip(
    shot: Shot,
    duration: float,
    resolution: tuple[int, int],
    params: VideoParams,
):
    """
    渲染压在画面上的字幕。

    强调字幕转成大写：这类短剧的钩子文字几乎都是全大写，小写会明显不像
    这个格式。台词和旁白保持原样，全大写的长句反而难读。
    """
    text = (shot.caption or shot.spoken_text()).strip()
    if not text:
        return None

    style = shot.caption_style
    if style == CaptionStyle.emphasis:
        text = text.upper()

    width, height = resolution
    font_path = utils.resolve_font_path(params.font_name)
    font_size = max(12, int(params.font_size * _CAPTION_SCALE.get(style, 1.0)))
    max_width = int(width * 0.86)

    wrapped, _ = video_service.wrap_text(
        text, max_width=max_width, font=font_path, fontsize=font_size
    )
    caption = TextClip(
        text=wrapped,
        font=font_path,
        font_size=font_size,
        color=params.text_fore_color,
        stroke_color=params.stroke_color,
        stroke_width=max(1, int(params.stroke_width)),
        method="label",
        text_align="center",
    ).with_duration(duration)

    vertical = _CAPTION_POSITION.get(style, 0.72)
    return caption.with_position(("center", int(height * vertical)))


def resolve_bgm_file(params: VideoParams, bgm_file_override: str | None) -> str:
    """
    决定这一集用哪段配乐。

    ``bgm_file_override`` 由任务层传入，表示配乐已由视频转音乐供应商按成片
    时长生成好；传入空串表示明确不要配乐。为 ``None`` 时才回退到内置随机
    曲库或用户上传的文件。
    """
    if bgm_file_override is not None:
        return bgm_file_override
    if not bgm_service.should_use_bgm(params.bgm_type, params.bgm_volume):
        return ""
    return video_service.get_bgm_file(
        bgm_type=params.bgm_type, bgm_file=params.bgm_file
    )


def assemble_episode(
    series: Series,
    episode: Episode,
    clips: Sequence[ClipResult],
    output_dir: str,
    params: VideoParams,
    bgm_file_override: str | None = None,
    output_name: str = "",
) -> str:
    """
    把画面、配音和字幕装配成一支成片，返回输出路径。

    镜头顺序就是叙事顺序，因此这里不做任何重排——素材库流程里的随机拼接
    在这条产线上是错误的。
    """
    if not clips:
        raise DramaAssemblyError("no shot visuals to assemble")
    shots = episode.shots()
    if len(shots) != len(clips):
        raise DramaAssemblyError(
            f"episode has {len(shots)} shots but {len(clips)} visuals were provided"
        )

    os.makedirs(output_dir, exist_ok=True)
    resolution = VideoAspect(params.video_aspect).to_resolution()
    audio_dir = os.path.join(output_dir, "audio")
    shot_audio = synthesize_shot_audio(series, episode, audio_dir, params)

    rendered = []
    opened = []
    try:
        for shot, clip_result, audio in zip(shots, clips, shot_audio):
            visual = build_shot_visual(
                clip_result, shot, audio.duration, resolution
            )
            opened.append(visual)

            layers = [visual]
            caption = build_caption_clip(shot, audio.duration, resolution, params)
            if caption is not None:
                opened.append(caption)
                layers.append(caption)

            composed = CompositeVideoClip(layers, size=resolution).with_duration(
                audio.duration
            )
            if audio.audio_path:
                voice_track = AudioFileClip(audio.audio_path)
                opened.append(voice_track)
                composed = composed.with_audio(voice_track)
            rendered.append(composed)

        logger.info(
            f"assembling episode: shots={len(rendered)}, "
            f"duration={sum(clip.duration for clip in rendered):.1f}s"
        )
        timeline = concatenate_videoclips(rendered, method="compose")
        opened.append(timeline)

        bgm_file = resolve_bgm_file(params, bgm_file_override)
        if bgm_file:
            try:
                effects = [
                    afx.MultiplyVolume(params.bgm_volume),
                    afx.AudioFadeOut(3),
                ]
                # 曲库里的歌通常比一集短，需要循环铺满；供应商按成片时长
                # 生成的配乐已经等长，再循环反而会在结尾多出一段。
                if bgm_file_override is None:
                    effects.append(afx.AudioLoop(duration=timeline.duration))
                music = AudioFileClip(bgm_file)
                opened.append(music)
                mixed = [music.with_effects(effects)]
                if timeline.audio is not None:
                    mixed.insert(0, timeline.audio)
                timeline = timeline.with_audio(CompositeAudioClip(mixed))
                logger.info(f"mixed background music: {os.path.basename(bgm_file)}")
            except Exception:
                # 配乐是锦上添花，混音失败不应当让整集白生成。记录完整堆栈
                # 供排查，成片继续按纯人声输出。
                logger.exception(f"failed to mix background music: {bgm_file}")

        name = output_name or f"episode-{episode.part_number:03d}"
        output_file = os.path.join(output_dir, f"{name}.mp4")
        video_service._write_videofile_with_codec_fallback(
            timeline,
            output_file,
            codec=video_service._get_effective_video_codec(),
            audio_codec="aac",
            fps=30,
            threads=params.n_threads or 2,
            logger=None,
        )
    finally:
        # 未关闭的 MoviePy 剪辑会持有文件句柄，Windows 上会让后续删除任务
        # 目录直接失败，因此无论成功与否都要收干净。
        for clip in opened:
            video_service.close_clip(clip)

    logger.success(f"episode assembled: {output_file}")
    return output_file
