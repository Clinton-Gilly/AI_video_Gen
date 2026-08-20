"""
API Key 与连通性测试页面。

配置散落在 config.toml 的多个小节里，而"这把 Key 到底能不能用"以前只能靠
真正跑一次任务来验证——任务跑到一半才因为 Key 无效失败，既浪费时间也浪费
已经花掉的额度。这一页把所有供应商集中在一处：填 Key、点 Connect、立刻看到
连接成功或者失败的具体原因。

所有探测都不触发计费生成，只做鉴权和元数据查询。
"""

import os
import sys

import streamlit as st

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
if root_dir in sys.path:
    sys.path.remove(root_dir)
sys.path.insert(0, root_dir)

from app.config import config  # noqa: E402
from app.models.llm_provider import LLM_PROVIDER_REGISTRY, get_llm_provider  # noqa: E402
from app.services import connections  # noqa: E402

st.set_page_config(page_title="Connections", page_icon="🔌", layout="wide")

st.title("🔌 Connections")
st.caption(
    "Enter each provider's API key and test it before running a job. "
    "No test triggers a billed generation."
)


def _render(result: connections.ConnectionResult) -> None:
    """统一渲染探测结果：成功给出用了多久，失败给出具体原因。"""
    if result.ok:
        st.success(f"**Connected** · {result.message} · {result.latency_ms} ms")
    else:
        st.error(f"**Failed** · {result.message}")


def _save(section: str, key: str, value: str) -> None:
    """把 Key 写回配置文件。留空表示不改动，避免误清已保存的值。"""
    if not value:
        return
    target = getattr(config, section, None)
    if target is None:
        st.warning(f"unknown config section: {section}")
        return
    target[key] = value
    config.save_config()
    st.toast(f"Saved {key}")


tab_llm, tab_visual, tab_music, tab_stock, tab_all = st.tabs(
    ["LLM", "Images & Video", "Music", "Stock footage", "Test everything"]
)


# --------------------------------------------------------------------------
# LLM —— 覆盖注册表里的全部供应商（含 Qwen、Ollama、Gemini 等）
# --------------------------------------------------------------------------
with tab_llm:
    st.subheader("Language model")
    st.caption(
        "The test sends a two-word prompt. That is the only probe that checks "
        "the key, base URL, model name and account balance at once."
    )

    provider_ids = [spec.provider_id for spec in LLM_PROVIDER_REGISTRY]
    current = str(config.app.get("llm_provider", "")).strip().lower()
    default_index = provider_ids.index(current) if current in provider_ids else 0

    provider_id = st.selectbox("Provider", provider_ids, index=default_index)
    spec = get_llm_provider(provider_id)

    saved_key = str(config.app.get(spec.config_key("api_key"), "") or "")
    saved_model = str(config.app.get(spec.config_key("model_name"), "") or "")
    saved_base = str(config.app.get(spec.config_key("base_url"), "") or "")

    col_key, col_model = st.columns(2)
    with col_key:
        api_key = st.text_input(
            "API key",
            value=saved_key,
            type="password",
            disabled=not spec.show_api_key,
            help=(
                "Not required for this provider"
                if not spec.requires_api_key
                else "Stored in config.toml when you press Save"
            ),
        )
    with col_model:
        model_name = st.text_input(
            "Model", value=saved_model, placeholder=spec.default_model
        )

    base_url = st.text_input(
        "Base URL",
        value=saved_base,
        disabled=not spec.show_base_url,
        placeholder="leave blank for the provider default",
    )

    extra_values = {}
    for extra_field in spec.extra_fields:
        extra_values[extra_field.config_suffix] = st.text_input(
            extra_field.config_suffix.replace("_", " ").title(),
            value=str(config.app.get(spec.config_key(extra_field.config_suffix), "") or ""),
        )

    col_connect, col_save = st.columns([1, 4])
    with col_connect:
        connect_llm = st.button("Connect", type="primary", key="connect_llm")
    with col_save:
        if st.button("Save to config.toml", key="save_llm"):
            config.app["llm_provider"] = provider_id
            _save("app", spec.config_key("api_key"), api_key)
            _save("app", spec.config_key("model_name"), model_name)
            _save("app", spec.config_key("base_url"), base_url)
            for suffix, value in extra_values.items():
                _save("app", spec.config_key(suffix), value)

    if connect_llm:
        with st.spinner(f"Testing {provider_id}…"):
            _render(
                connections.test_llm(
                    provider_id=provider_id,
                    api_key=api_key,
                    model_name=model_name,
                    base_url=base_url,
                    extra=extra_values,
                )
            )


# --------------------------------------------------------------------------
# 画面：静帧与图生视频
# --------------------------------------------------------------------------
with tab_visual:
    col_stills, col_motion = st.columns(2)

    with col_stills:
        st.subheader("Stills — Gemini")
        st.caption("Tested by listing models, so no image is generated or billed.")
        gemini_key = st.text_input(
            "Gemini API key",
            value=str(
                config.app.get("gemini_image_api_key")
                or config.app.get("gemini_api_key")
                or ""
            ),
            type="password",
            help="Shared with the LLM section when gemini_image_api_key is unset.",
        )
        gemini_model = st.text_input(
            "Image model",
            value=str(config.app.get("gemini_image_model", "") or ""),
            placeholder="gemini-3-pro-image-preview",
        )
        if st.button("Connect", type="primary", key="connect_stills"):
            with st.spinner("Testing Gemini…"):
                _render(connections.test_still_provider(gemini_key, gemini_model))
        if st.button("Save", key="save_stills"):
            _save("app", "gemini_image_api_key", gemini_key)
            _save("app", "gemini_image_model", gemini_model)

    with col_motion:
        st.subheader("Motion — Kling")
        st.caption(
            "Tested by querying a task id that does not exist. A 'not found' "
            "reply proves the key authenticated; nothing is generated or billed."
        )
        kling_key = st.text_input(
            "Kling API key",
            value=str(config.app.get("kling_api_key", "") or ""),
            type="password",
        )
        kling_base = st.text_input(
            "Base URL",
            value=str(config.app.get("kling_base_url", "") or ""),
            placeholder="https://api.klingai.com",
        )
        if st.button("Connect", type="primary", key="connect_motion"):
            with st.spinner("Testing Kling…"):
                _render(connections.test_motion_provider(kling_key, kling_base))
        if st.button("Save", key="save_motion"):
            _save("app", "kling_api_key", kling_key)
            _save("app", "kling_base_url", kling_base)


# --------------------------------------------------------------------------
# 配乐
# --------------------------------------------------------------------------
with tab_music:
    st.subheader("Background music")
    st.caption(
        "These providers watch the finished episode and score it. Both expose "
        "a free account endpoint, which is what the test calls."
    )

    col_sonilo, col_eleven = st.columns(2)
    with col_sonilo:
        st.markdown("**Sonilo**")
        sonilo_key = st.text_input(
            "Sonilo API key",
            value=str(config.app.get("sonilo_api_key", "") or ""),
            type="password",
        )
        if st.button("Connect", type="primary", key="connect_sonilo"):
            with st.spinner("Testing Sonilo…"):
                _render(connections.test_music_provider("sonilo"))
        if st.button("Save", key="save_sonilo"):
            _save("app", "sonilo_api_key", sonilo_key)

    with col_eleven:
        st.markdown("**ElevenLabs**")
        eleven_key = st.text_input(
            "ElevenLabs API key",
            value=str(config.elevenlabs.get("api_key", "") or ""),
            type="password",
        )
        if st.button("Connect", type="primary", key="connect_eleven"):
            with st.spinner("Testing ElevenLabs…"):
                _render(connections.test_music_provider("elevenlabs"))
        if st.button("Save", key="save_eleven"):
            _save("elevenlabs", "api_key", eleven_key)

    st.info(
        "Drama episodes also accept the built-in song library — set "
        "`bgm_type` to `random` and no key is needed."
    )


# --------------------------------------------------------------------------
# 素材库（仅 stock 模式使用）
# --------------------------------------------------------------------------
with tab_stock:
    st.subheader("Stock footage")
    st.caption("Used by stock mode only. Character drama does not need these.")

    for provider in ("pexels", "pixabay"):
        keys = config.app.get(f"{provider}_api_keys") or []
        if isinstance(keys, str):
            keys = [keys]
        st.markdown(f"**{provider.title()}**")
        value = st.text_input(
            f"{provider} API key",
            value=str(keys[0]) if keys else "",
            type="password",
            key=f"stock_key_{provider}",
        )
        col_connect, col_save = st.columns([1, 4])
        with col_connect:
            if st.button("Connect", type="primary", key=f"connect_{provider}"):
                with st.spinner(f"Testing {provider}…"):
                    _render(connections.test_stock_provider(provider, value))
        with col_save:
            if st.button("Save", key=f"save_{provider}"):
                if value:
                    config.app[f"{provider}_api_keys"] = [value]
                    config.save_config()
                    st.toast(f"Saved {provider} key")
        st.divider()


# --------------------------------------------------------------------------
# 一次性检查全部
# --------------------------------------------------------------------------
with tab_all:
    st.subheader("Test everything that is configured")
    st.caption(
        "Providers with no key are skipped rather than reported as failures, "
        "so a real error does not get lost in a wall of red."
    )
    if st.button("Run all tests", type="primary", key="connect_all"):
        with st.spinner("Testing every configured provider…"):
            results = connections.test_all()
        ok_count = sum(1 for entry in results if entry.ok)
        st.write(f"**{ok_count} of {len(results)} connected**")
        for entry in results:
            st.markdown(f"**{entry.name}**")
            _render(entry)
