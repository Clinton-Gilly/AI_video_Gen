import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cli
from app.models.const import TASK_STATE_COMPLETE, TASK_STATE_FAILED
from app.models.schema import VideoMode, VideoParams
from app.models.story import Character, Episode, Scene, Series, Shot, StoryBeat
from app.services import series as series_store
from app.services import task
from app.utils import utils


def _series(**overrides) -> Series:
    payload = {
        "id": "fruit-court",
        "title": "Fruit Court",
        "visual_style": "soft studio lighting",
        "cast": [
            Character(
                id="berry",
                name="Berry",
                head_type="strawberry",
                reference_image="storage/series/fruit-court/cast/berry.png",
            )
        ],
    }
    payload.update(overrides)
    return Series(**payload)


def _episode(part_number: int = 1) -> Episode:
    shot = Shot(
        index=1,
        action="slams a folder onto the table",
        character_ids=["berry"],
        narration="She had waited three weeks.",
        duration=4.0,
    )
    return Episode(
        series_id="fruit-court",
        part_number=part_number,
        title=f"Part {part_number}",
        scenes=[
            Scene(index=1, beat=StoryBeat.hook, setting="a courtroom", shots=[shot])
        ],
        cliffhanger="who signed it?",
    )


class TestModeRouting(unittest.TestCase):
    """模式选择必须真正改变产线，而不是只改一个字段。"""

    def test_stock_is_the_default_so_existing_callers_are_unaffected(self):
        self.assertEqual(VideoParams(video_subject="anything").mode, VideoMode.stock)

    def test_default_params_route_to_the_stock_pipeline(self):
        params = VideoParams(video_subject="how AI helps developers")
        with patch.object(task, "_run_stock_pipeline") as stock, patch.object(
            task, "_run_drama_pipeline"
        ) as drama:
            task._run_pipeline("task-1", params)
        stock.assert_called_once()
        drama.assert_not_called()

    def test_drama_params_route_to_the_drama_pipeline(self):
        params = VideoParams(
            video_subject="a disputed receipt",
            mode=VideoMode.drama,
            series_id="fruit-court",
        )
        with patch.object(task, "_run_stock_pipeline") as stock, patch.object(
            task, "_run_drama_pipeline"
        ) as drama:
            task._run_pipeline("task-1", params)
        drama.assert_called_once()
        stock.assert_not_called()


class TestDramaPipeline(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.storage = Path(self.temp_dir.name)

        def fake_storage_dir(sub_dir: str = "", create: bool = False) -> str:
            target = self.storage / sub_dir if sub_dir else self.storage
            if create:
                target.mkdir(parents=True, exist_ok=True)
            return str(target)

        self.patches = [
            patch("app.services.series.utils.storage_dir", side_effect=fake_storage_dir),
            patch.object(utils, "task_dir", return_value=str(self.storage / "task")),
        ]
        for entry in self.patches:
            entry.start()
        (self.storage / "task").mkdir(parents=True, exist_ok=True)
        self.states = {}
        self.state_patch = patch.object(
            task.sm.state, "update_task", side_effect=self._record_state
        )
        self.state_patch.start()

    def _record_state(self, task_id, state=None, progress=0, **kwargs):
        self.states[task_id] = {"state": state, "progress": progress, **kwargs}

    def tearDown(self):
        self.state_patch.stop()
        for entry in self.patches:
            entry.stop()
        self.temp_dir.cleanup()

    def _params(self, **overrides) -> VideoParams:
        payload = {
            "video_subject": "a disputed receipt",
            "mode": VideoMode.drama,
            "series_id": "fruit-court",
        }
        payload.update(overrides)
        return VideoParams(**payload)

    def test_missing_series_id_fails_before_any_llm_call(self):
        with patch("app.services.series.continue_series") as generate:
            result = task._run_drama_pipeline("t1", self._params(series_id=None))
        generate.assert_not_called()
        self.assertEqual(self.states["t1"]["state"], TASK_STATE_FAILED)
        self.assertIn("series_id", result["error"])

    def test_unknown_series_fails_before_any_llm_call(self):
        with patch("app.services.series.continue_series") as generate:
            result = task._run_drama_pipeline("t1", self._params())
        generate.assert_not_called()
        self.assertEqual(self.states["t1"]["state"], TASK_STATE_FAILED)
        self.assertIn("series not found", result["error"])

    def test_stop_at_script_completes_without_needing_a_visual_backend(self):
        """剧本阶段不消耗画面额度，没有画面后端也应当能产出剧本。"""
        series_store.save_series(_series())
        with patch("app.services.series.continue_series", return_value=_episode(1)):
            result = task._run_drama_pipeline("t1", self._params(), stop_at="script")

        self.assertEqual(self.states["t1"]["state"], TASK_STATE_COMPLETE)
        self.assertEqual(result["shot_count"], 1)
        self.assertEqual(result["part_number"], 1)
        self.assertEqual(result["cliffhanger"], "who signed it?")

    def test_characters_without_a_reference_image_block_rendering(self):
        """缺定妆图的角色会逐镜换脸，必须在生成画面之前拦下。"""
        series_store.save_series(
            _series(cast=[Character(id="berry", name="Berry", head_type="strawberry")])
        )
        with patch("app.services.series.continue_series", return_value=_episode(1)):
            result = task._run_drama_pipeline("t1", self._params())

        self.assertEqual(self.states["t1"]["state"], TASK_STATE_FAILED)
        self.assertIn("berry", result["error"])
        self.assertIn("reference image", result["error"])

    def test_drama_never_silently_falls_back_to_stock_footage(self):
        """
        角色短剧退回素材库空镜就变成了另一种视频。画面后端缺失时必须明确失败。
        """
        series_store.save_series(_series())
        with patch(
            "app.services.series.continue_series", return_value=_episode(1)
        ), patch.object(task, "get_video_materials") as materials:
            result = task._run_drama_pipeline("t1", self._params())

        materials.assert_not_called()
        self.assertEqual(self.states["t1"]["state"], TASK_STATE_FAILED)
        # 没有配置画面后端时报告的是后端缺失，而不是悄悄改用素材库空镜。
        self.assertIn("not configured", result["error"])

    def test_episode_premise_takes_precedence_over_the_subject(self):
        series_store.save_series(_series())
        with patch(
            "app.services.series.continue_series", return_value=_episode(1)
        ) as generate:
            task._run_drama_pipeline(
                "t1",
                self._params(episode_premise="the signature is examined"),
                stop_at="script",
            )
        self.assertEqual(
            generate.call_args.kwargs["premise"], "the signature is examined"
        )


class TestCliModeSelection(unittest.TestCase):
    def test_drama_mode_requires_a_series(self):
        with self.assertRaises(SystemExit):
            cli.parse_args(["--mode", "drama", "--episode-premise", "x"])

    def test_drama_flags_are_rejected_outside_drama_mode(self):
        """静默忽略会让用户以为设置生效了。"""
        with self.assertRaises(SystemExit):
            cli.parse_args(["--video-subject", "x", "--series-id", "s"])

    def test_episode_premise_satisfies_the_content_requirement(self):
        args = cli.parse_args(
            [
                "--mode",
                "drama",
                "--series-id",
                "fruit-court",
                "--episode-premise",
                "a disputed receipt",
            ]
        )
        self.assertEqual(args.mode, "drama")
        params = cli.build_video_params(args)
        self.assertEqual(params.mode, VideoMode.drama)
        self.assertEqual(params.series_id, "fruit-court")

    def test_stock_remains_the_default_from_the_command_line(self):
        args = cli.parse_args(["--video-subject", "how AI helps"])
        params = cli.build_video_params(args)
        self.assertEqual(params.mode, VideoMode.stock)


if __name__ == "__main__":
    unittest.main()
