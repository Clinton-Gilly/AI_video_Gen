"""
角色短剧的独立页面。

刻意不写进 ``webui/Main.py``：那个文件已经接近五千行，专门服务素材库流程的
参数面板。短剧的操作对象是系列、角色和分集，与素材、关键词、转场几乎没有
共用控件，塞进同一页只会让两条产线互相干扰。Streamlit 的多页机制让这一页
独立存在，Main.py 完全不受影响。

这一页的职责是"看得见"：建系列、看角色定妆图、写分集、看每一镜的画面，
以及在真正花钱之前先看到这一集会花多少。
"""

import os
import sys

import streamlit as st

# 与 Main.py 相同的路径处理：确保项目自己的 app 包优先于第三方同名包。
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
if root_dir in sys.path:
    sys.path.remove(root_dir)
sys.path.insert(0, root_dir)

from app.config import config  # noqa: E402
from app.models.schema import VideoAspect, VideoMode, VideoParams  # noqa: E402
from app.services import series as series_store  # noqa: E402
from app.services import task as task_service  # noqa: E402
from app.services import visual  # noqa: E402
from app.services import voice as voice_service  # noqa: E402
from app.utils import utils  # noqa: E402

st.set_page_config(
    page_title="Character Drama", page_icon="🎭", layout="wide"
)

st.title("🎭 Character Drama")
st.caption(
    "Generate recurring-cast mini-dramas: a locked cast, per-shot visuals, "
    "and Part 1 / Part 2 cliffhangers."
)


def _series_options() -> list[str]:
    try:
        return series_store.list_series()
    except Exception as exc:  # pragma: no cover - defensive UI guard
        st.error(f"failed to list series: {exc}")
        return []


def _provider_status() -> tuple[bool, bool]:
    try:
        stills = visual.get_still_provider().is_enabled()
    except Exception:
        stills = False
    try:
        motion = visual.get_motion_provider().is_enabled()
    except Exception:
        motion = False
    return stills, motion


stills_ready, motion_ready = _provider_status()

with st.sidebar:
    st.subheader("Backends")
    st.write(f"{'✅' if stills_ready else '❌'} stills — `{visual.DEFAULT_STILL_PROVIDER}`")
    st.write(f"{'✅' if motion_ready else '❌'} motion — `{visual.DEFAULT_MOTION_PROVIDER}`")
    if not stills_ready:
        st.warning("Set `gemini_api_key` in config.toml to generate images.")
    if not motion_ready:
        st.info(
            "Without a motion backend you can still render every shot as a "
            "still with a Ken Burns move. Set `max_animated_shots` to 0."
        )

tab_series, tab_cast, tab_episode = st.tabs(
    ["1 · Series", "2 · Cast", "3 · Episode"]
)


# --------------------------------------------------------------------------
# 1. 系列：风格圣经和常驻角色
# --------------------------------------------------------------------------
with tab_series:
    st.subheader("Create a series")
    st.caption(
        "The series holds what never changes: the visual style and the "
        "recurring cast. Every episode inherits it."
    )

    with st.form("create_series"):
        col_left, col_right = st.columns(2)
        with col_left:
            new_series_id = st.text_input(
                "Series id", value="", placeholder="fruit-court",
                help="Lowercase letters, digits, hyphens. Used as the folder name.",
            )
            concept = st.text_area(
                "Concept",
                placeholder="Neighbours settle petty disputes in a tiny fruit courtroom",
                height=90,
            )
            character_kind = st.text_input("Character heads", value="fruit")
        with col_right:
            cast_size = st.slider("Cast size", 1, 8, 3)
            aspect = st.selectbox("Aspect", ["9:16", "16:9", "1:1"], index=0)
            language = st.text_input("Language", value="", placeholder="en-US")
            custom_style = st.text_area(
                "Visual style (optional)",
                placeholder="Leave blank to let the model design one",
                height=90,
            )

        if st.form_submit_button("Create series", type="primary"):
            if not new_series_id.strip() or not concept.strip():
                st.error("Series id and concept are both required.")
            else:
                with st.spinner("Writing the series bible…"):
                    try:
                        created = series_store.create_series(
                            concept=concept,
                            series_id=new_series_id.strip(),
                            cast_size=cast_size,
                            language=language,
                            character_kind=character_kind,
                            custom_style=custom_style,
                            aspect=VideoAspect(aspect),
                        )
                        st.success(f"Created **{created.title}**")
                        st.rerun()
                    except Exception as exc:
                        st.error(str(exc))

    existing = _series_options()
    if existing:
        st.divider()
        st.subheader("Existing series")
        for series_id in existing:
            try:
                entry = series_store.load_series(series_id)
            except Exception as exc:
                st.warning(f"{series_id}: {exc}")
                continue
            parts = series_store.list_episodes(series_id)
            with st.expander(f"**{entry.title}** · `{series_id}` · {len(parts)} parts"):
                st.write(entry.premise or "_no premise_")
                if entry.visual_style:
                    st.caption(f"Style: {entry.visual_style}")
                st.write(
                    "Cast: " + ", ".join(f"`{m.id}`" for m in entry.cast)
                )
                missing = entry.missing_references()
                if missing:
                    st.warning(
                        "No reference image yet: " + ", ".join(missing)
                    )


# --------------------------------------------------------------------------
# 2. 角色：定妆图和音色
# --------------------------------------------------------------------------
with tab_cast:
    options = _series_options()
    if not options:
        st.info("Create a series first.")
    else:
        selected = st.selectbox("Series", options, key="cast_series")
        series = series_store.load_series(selected)

        st.caption(
            "Reference images are the consistency anchor: every shot is "
            "conditioned on them, so a character keeps the same face across "
            "shots and across episodes."
        )

        missing = series.missing_references()
        if missing and stills_ready:
            if st.button(f"Generate {len(missing)} missing reference image(s)"):
                with st.spinner("Generating reference images…"):
                    try:
                        cast_dir = os.path.join(
                            str(series_store.series_dir(selected)), "cast"
                        )
                        visual.generate_reference_images(series, cast_dir)
                        st.success("Reference images ready.")
                        st.rerun()
                    except Exception as exc:
                        st.error(str(exc))
        elif missing:
            st.warning(
                "Configure a still backend to generate the missing reference "
                "images: " + ", ".join(missing)
            )

        try:
            voices = voice_service.get_all_azure_voices(filter_locals=None)
        except Exception:
            voices = []

        columns = st.columns(min(len(series.cast), 4) or 1)
        for index, member in enumerate(series.cast):
            with columns[index % len(columns)]:
                reference = series_store.reference_image_path(series, member.id)
                if reference is not None:
                    st.image(str(reference), caption=member.name)
                else:
                    st.info(f"{member.name} — no reference image")
                st.caption(f"`{member.id}` · {member.head_type or 'human'}")
                if member.appearance:
                    st.caption(member.appearance)

                # 每个角色可以有自己的音色，这是多角色短剧听起来像对话而不是
                # 一个人念稿的关键。
                current = member.voice_name
                choices = [""] + voices
                picked = st.selectbox(
                    "Voice",
                    choices,
                    index=choices.index(current) if current in choices else 0,
                    key=f"voice_{selected}_{member.id}",
                    format_func=lambda v: v or "— use narrator voice —",
                )
                if picked != current:
                    member.voice_name = picked
                    series_store.save_series(series)
                    st.toast(f"Voice saved for {member.name}")


# --------------------------------------------------------------------------
# 3. 分集：写剧本、看方案与成本、生成
# --------------------------------------------------------------------------
with tab_episode:
    options = _series_options()
    if not options:
        st.info("Create a series first.")
    else:
        selected = st.selectbox("Series", options, key="episode_series")
        series = series_store.load_series(selected)
        next_part = series.next_part_number()

        previous = series_store.previous_cliffhanger(selected, next_part)
        if previous:
            st.info(f"Part {next_part} must pay off: _{previous}_")

        premise = st.text_area(
            f"What happens in Part {next_part}?",
            placeholder="A disputed receipt turns into a shouting match",
            height=90,
        )
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            is_finale = st.checkbox("Final part (resolve, no cliffhanger)")
        with col_b:
            max_animated = st.slider(
                "Animated shots",
                0, 12, int(config.app.get("max_animated_shots", 4)),
                help=(
                    "Shots given real image-to-video motion. Everything else "
                    "gets a free Ken Burns move. This is the main cost lever."
                ),
            )
        with col_c:
            stop_at = st.selectbox(
                "Stop after", ["script", "materials", "video"], index=0,
                help="Script costs only an LLM call and spends nothing on images.",
            )

        if st.button("Generate", type="primary", disabled=not premise.strip()):
            params = VideoParams(
                video_subject=premise,
                mode=VideoMode.drama,
                series_id=selected,
                episode_premise=premise,
                is_finale=is_finale,
                video_aspect=series.aspect,
                max_animated_shots=max_animated,
            )
            task_id = utils.get_uuid()
            st.session_state["drama_task_id"] = task_id
            with st.spinner(f"Running drama pipeline (stop at {stop_at})…"):
                result = task_service.start(task_id, params, stop_at=stop_at)
            if result and not result.get("error"):
                st.success(f"Done — task `{task_id}`")
                st.session_state["drama_result"] = result
            else:
                st.error((result or {}).get("error", "task failed"))
                st.session_state["drama_result"] = None

        result = st.session_state.get("drama_result")
        if result:
            st.divider()
            st.subheader(result.get("title", "Episode"))
            if result.get("hook"):
                st.caption(f"Hook: {result['hook']}")
            if result.get("cliffhanger"):
                st.caption(f"Cliffhanger: {result['cliffhanger']}")
            st.write(
                f"{result.get('shot_count', 0)} shots · "
                f"~{result.get('estimated_duration', 0):.0f}s · "
                f"{result.get('animated_shot_count', 0)} animated"
            )

            videos = result.get("videos") or []
            if videos and os.path.exists(videos[0]):
                st.video(videos[0])

            stills = result.get("stills") or []
            if stills:
                st.subheader("Shots")
                gallery = st.columns(4)
                for index, still in enumerate(stills):
                    if not os.path.exists(still):
                        continue
                    with gallery[index % 4]:
                        st.image(still, caption=f"shot {index + 1}")

            clips = result.get("animated_clips") or []
            if clips:
                st.subheader("Animated shots")
                clip_columns = st.columns(min(len(clips), 3))
                for index, clip in enumerate(clips):
                    if not os.path.exists(clip):
                        continue
                    with clip_columns[index % len(clip_columns)]:
                        st.video(clip)
