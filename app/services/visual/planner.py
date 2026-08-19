"""
决定每个镜头是调用图生视频模型，还是只出静帧靠 Ken Burns 制造运动。

单集成本几乎全部来自图生视频：静帧按张计费且单价低，视频按秒计费且单价高
一个数量级。因此控制成本最有效的手段不是换更便宜的模型，而是**少动几个
镜头**——竖屏短剧里，一个静止的对话特写配上缓慢推镜，观众在手机上分辨不出
它没有真正动过。

按经验，一集只让 3-5 个关键镜头真正动起来，成本可以降到全动方案的三分之一
左右，观感差异却很小。这里把"哪些镜头值得动"变成可测试的打分规则，而不是
交给调用方临时判断。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from loguru import logger

from app.models.story import CameraMove, Episode, Shot, ShotSize, StoryBeat


# 默认预算：一集最多动 4 个镜头。观感上足够撑住节奏，成本约为全动方案的
# 三分之一。设为 0 表示整集都用 Ken Burns，设为很大的值则等于全动。
DEFAULT_MAX_ANIMATED_SHOTS = 4

# 叙事节拍权重。钩子决定完播率，转折和悬念决定观众是否等下一集，这三处的
# 真实运动最值钱；铺垫和交代类镜头即使静止也不影响理解。
_BEAT_SCORES = {
    StoryBeat.hook: 5.0,
    StoryBeat.turn: 4.0,
    StoryBeat.cliffhanger: 4.0,
    StoryBeat.inciting_incident: 3.0,
    StoryBeat.escalation: 2.5,
    StoryBeat.consequence: 2.0,
    StoryBeat.resolution: 1.5,
    StoryBeat.setup: 0.5,
}

# 镜别权重。特写里有面部表情和口型，动起来收益最大；插入镜头拍的是静物，
# 真实运动几乎看不出来，是最该省下的地方。
_SHOT_SIZE_SCORES = {
    ShotSize.extreme_close_up: 2.0,
    ShotSize.close_up: 2.0,
    ShotSize.over_the_shoulder: 1.5,
    ShotSize.medium: 1.0,
    ShotSize.wide: 0.5,
    ShotSize.establishing: 0.5,
    ShotSize.insert: -1.0,
}


@dataclass(frozen=True)
class ShotPlan:
    """单个镜头的画面方案。"""

    shot: Shot
    beat: StoryBeat
    animate: bool
    score: float
    reason: str


def score_shot(shot: Shot, beat: StoryBeat) -> float:
    """给镜头打分，分数越高越值得花钱做真实运动。"""
    score = _BEAT_SCORES.get(beat, 1.0)
    score += _SHOT_SIZE_SCORES.get(shot.shot_size, 1.0)

    # 有台词的镜头需要口型和头部动作，静帧最容易露馅。
    if shot.dialogue:
        score += 2.0

    # 剧本显式要求了运镜，说明这一镜的设计依赖镜头移动。Ken Burns 只能做
    # 推拉和平移，手持等运动无法伪造，权重更高。
    if shot.camera_move == CameraMove.handheld:
        score += 1.5
    elif shot.camera_move != CameraMove.static:
        score += 0.5

    return score


def plan_episode(
    episode: Episode,
    max_animated_shots: int = DEFAULT_MAX_ANIMATED_SHOTS,
) -> List[ShotPlan]:
    """
    为一集里的每个镜头决定画面方案。

    先按分数选出预算内最值得动的镜头，再按镜头顺序返回，方便调用方直接顺序
    执行。分数相同时保留剧本顺序，保证同一份剧本每次得到相同的方案——方案
    不稳定会让成本核算和重跑结果无法比较。
    """
    if max_animated_shots < 0:
        raise ValueError("max_animated_shots cannot be negative")

    scored: list[tuple[float, int, Shot, StoryBeat]] = []
    for scene in episode.scenes:
        for shot in scene.shots:
            scored.append((score_shot(shot, scene.beat), len(scored), shot, scene.beat))

    # 先按分数降序，再按出现顺序升序，保证结果稳定可复现。
    ranked = sorted(scored, key=lambda entry: (-entry[0], entry[1]))
    animated_positions = {entry[1] for entry in ranked[:max_animated_shots]}

    plans = [
        ShotPlan(
            shot=shot,
            beat=beat,
            animate=position in animated_positions,
            score=score,
            reason=(
                f"{beat.value} beat, {shot.shot_size.value}"
                + (", has dialogue" if shot.dialogue else "")
            ),
        )
        for score, position, shot, beat in scored
    ]

    animated = sum(1 for plan in plans if plan.animate)
    animated_seconds = sum(plan.shot.duration for plan in plans if plan.animate)
    logger.info(
        f"visual plan: shots={len(plans)}, animated={animated}, "
        f"animated_seconds={animated_seconds:.1f}, "
        f"still_only={len(plans) - animated}"
    )
    return plans


def estimate_cost(
    plans: List[ShotPlan],
    cost_per_video_second: float,
    cost_per_still: float = 0.0,
) -> dict:
    """
    估算一集的画面成本，供调用方在真正生成之前决定是否继续。

    单价由调用方按当前供应商传入，本模块不内置价格——各家价格变动频繁，
    写死在代码里很快就会误导使用者。
    """
    if cost_per_video_second < 0 or cost_per_still < 0:
        raise ValueError("costs cannot be negative")

    animated_seconds = sum(plan.shot.duration for plan in plans if plan.animate)
    still_count = len(plans)
    video_cost = animated_seconds * cost_per_video_second
    still_cost = still_count * cost_per_still

    # 全动方案作为对照，让"少动几个镜头"省下的钱是一个具体数字。
    full_motion_seconds = sum(plan.shot.duration for plan in plans)
    full_motion_cost = full_motion_seconds * cost_per_video_second + still_cost

    return {
        "shot_count": still_count,
        "animated_shots": sum(1 for plan in plans if plan.animate),
        "animated_seconds": animated_seconds,
        "still_cost": still_cost,
        "video_cost": video_cost,
        "total_cost": video_cost + still_cost,
        "full_motion_cost": full_motion_cost,
        "saved": full_motion_cost - (video_cost + still_cost),
    }
