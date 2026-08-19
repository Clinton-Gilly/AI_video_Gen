"""任务目录中持久化文件的安全读写。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from loguru import logger

from app.utils import utils


def _script_file(task_id: str) -> Path:
    """返回任务脚本清单路径，并复用统一的任务目录创建逻辑。"""
    return Path(utils.task_dir(task_id)) / "script.json"


def _write_json_atomic(target: Path, payload: Mapping[str, Any]) -> None:
    """原子写入任务清单，实现与剧集、角色档案共享同一份落盘语义。"""
    utils.write_json_atomic(target, payload)


def write_script_data(task_id: str, payload: Mapping[str, Any]) -> None:
    """创建或完整替换任务的 ``script.json`` 清单。"""
    _write_json_atomic(_script_file(task_id), payload)


def patch_script_data(task_id: str, **updates: Any) -> bool:
    """
    在保留原有字段的前提下补充任务清单，失败时返回 ``False``。

    素材来源属于辅助诊断信息，不能因为文件权限、磁盘瞬时异常或历史文件损坏
    阻断视频生成。因此该入口会记录完整异常并降级；首次创建任务清单仍使用
    ``write_script_data``，由主流程决定基础任务数据写入失败时如何处理。
    """
    try:
        target = _script_file(task_id)
        with target.open("r", encoding="utf-8") as script_file:
            payload = json.load(script_file)
        if not isinstance(payload, dict):
            raise ValueError("task script data must be a JSON object")

        payload.update(updates)
        _write_json_atomic(target, payload)
        return True
    except FileNotFoundError:
        # ``download_videos`` 也可能被测试、脚本或第三方代码独立调用，此时没有
        # 任务清单属于正常场景，不应制造警告或为了辅助记录创建残缺文件。
        logger.debug(
            f"skip task script update because script.json does not exist: "
            f"task_id={task_id}"
        )
        return False
    except Exception as exc:
        logger.warning(
            "failed to update task script data: "
            f"task_id={task_id}, fields={sorted(updates)}, "
            f"error={type(exc).__name__}, detail={exc}"
        )
        return False
