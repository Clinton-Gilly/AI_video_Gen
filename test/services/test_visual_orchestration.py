import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from app.models.story import (
    Character,
    Episode,
    Scene,
    Series,
    Shot,
    ShotSize,
    StoryBeat,
)
from app.services import series as series_store
from app.services import visual
from app.services.visual.base import StillResult, VisualConfigurationError


def _character(character_id: str, with_reference: bool = True) -> Character:
    return Character(
        id=character_id,
        name=character_id.title(),
        head_type="strawberry",
        appearance="red hoodie",
        reference_image=(
            f"series/fruit-court/cast/{character_id}.png"
            if with_reference
            else ""
        ),
    )


def _series(**overrides) -> Series:
    payload = {
        "id": "fruit-court",
        "title": "Fruit Court",
        "visual_style": "soft studio lighting",
        "cast": [_character("berry"), _character("mango")],
    }
    payload.update(overrides)
    return Series(**payload)


def _episode() -> Episode:
    return Episode(
        series_id="fruit-court",
        title="The Missing Receipt",
        scenes=[
            Scene(
                index=1,
                beat=StoryBeat.hook,
                setting="a courtroom",
                shots=[
                    Shot(
                        index=1,
                        action="slams a folder down",
                        character_ids=["berry"],
                        shot_size=ShotSize.close_up,
                        dialogue={"speaker_id": "berry", "line": "I kept it."},
                        duration=3.0,
                    ),
                    Shot(
                        index=2,
                        action="a receipt lies on the table",
                        character_ids=[],
                        shot_size=ShotSize.insert,
                        narration="The paper said otherwise.",
                        duration=2.0,
                    ),
                ],
            )
        ],
        cliffhanger="who signed it?",
    )


def _still_provider(enabled: bool = True) -> MagicMock:
    provider = MagicMock()
    provider.name = "fake-stills"
    provider.is_enabled.return_value = enabled
    provider.generate_still.side_effect = lambda **kwargs: StillResult(
        path=kwargs["output_path"], provider="fake-stills", model="fake"
    )
    return provider


def _motion_provider(enabled: bool = True) -> MagicMock:
    provider = MagicMock()
    provider.name = "fake-motion"
    provider.is_enabled.return_value = enabled
    provider.animate.side_effect = lambda **kwargs: kwargs["output_path"]
    return provider


class VisualTestCase(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.storage = Path(self.temp_dir.name)
        self.output = self.storage / "shots"

        def fake_storage_dir(sub_dir: str = "", create: bool = False) -> str:
            target = self.storage / sub_dir if sub_dir else self.storage
            if create:
                target.mkdir(parents=True, exist_ok=True)
            return str(target)

        self.storage_patch = patch(
            "app.services.series.utils.storage_dir", side_effect=fake_storage_dir
        )
        self.storage_patch.start()

    def tearDown(self):
        self.storage_patch.stop()
        self.temp_dir.cleanup()


class TestEpisodeVisuals(VisualTestCase):
    def test_generates_one_still_per_shot(self):
        stills = _still_provider()
        visual.generate_episode_visuals(
            series=_series(),
            episode=_episode(),
            output_dir=str(self.output),
            still_provider=stills,
            motion_provider=_motion_provider(),
            max_animated_shots=1,
        )
        self.assertEqual(stills.generate_still.call_count, 2)

    def test_only_planned_shots_are_animated(self):
        """图生视频按秒计费，多动一个镜头就是多花一份钱。"""
        motion = _motion_provider()
        results = visual.generate_episode_visuals(
            series=_series(),
            episode=_episode(),
            output_dir=str(self.output),
            still_provider=_still_provider(),
            motion_provider=motion,
            max_animated_shots=1,
        )
        self.assertEqual(motion.animate.call_count, 1)
        self.assertEqual([r.animated for r in results], [True, False])
        self.assertIsNone(results[1].clip_path)

    def test_a_zero_budget_never_touches_the_motion_provider(self):
        motion = _motion_provider()
        results = visual.generate_episode_visuals(
            series=_series(),
            episode=_episode(),
            output_dir=str(self.output),
            still_provider=_still_provider(),
            motion_provider=motion,
            max_animated_shots=0,
        )
        motion.animate.assert_not_called()
        self.assertTrue(all(not r.animated for r in results))

    def test_shot_stills_carry_the_reference_images_of_who_is_on_screen(self):
        """参考图是跨镜头一致性的条件输入，漏传角色就会换脸。"""
        (self.storage / "series" / "fruit-court" / "cast").mkdir(parents=True)
        (self.storage / "series" / "fruit-court" / "cast" / "berry.png").write_bytes(
            b"png"
        )
        stills = _still_provider()
        visual.generate_episode_visuals(
            series=_series(),
            episode=_episode(),
            output_dir=str(self.output),
            still_provider=stills,
            motion_provider=_motion_provider(),
            max_animated_shots=0,
        )
        first_call = stills.generate_still.call_args_list[0].kwargs
        self.assertEqual(len(first_call["reference_images"]), 1)
        self.assertTrue(first_call["reference_images"][0].endswith("berry.png"))
        # 插入镜头没有角色出场，不该带任何参考图。
        second_call = stills.generate_still.call_args_list[1].kwargs
        self.assertEqual(list(second_call["reference_images"]), [])

    def test_missing_reference_images_stop_generation_before_any_call(self):
        """先画完整集再发现要重做，代价远高于在这里停下。"""
        stills = _still_provider()
        with self.assertRaises(VisualConfigurationError) as context:
            visual.generate_episode_visuals(
                series=_series(cast=[_character("berry", with_reference=False)]),
                episode=_episode(),
                output_dir=str(self.output),
                still_provider=stills,
                motion_provider=_motion_provider(),
            )
        stills.generate_still.assert_not_called()
        self.assertIn("berry", str(context.exception))

    def test_an_unconfigured_motion_provider_fails_before_spending_on_stills(self):
        """静帧也要花钱，画完再发现动不了等于白花。"""
        stills = _still_provider()
        with self.assertRaises(VisualConfigurationError) as context:
            visual.generate_episode_visuals(
                series=_series(),
                episode=_episode(),
                output_dir=str(self.output),
                still_provider=stills,
                motion_provider=_motion_provider(enabled=False),
                max_animated_shots=2,
            )
        stills.generate_still.assert_not_called()
        self.assertIn("max_animated_shots to 0", str(context.exception))

    def test_an_unconfigured_still_provider_is_rejected(self):
        with self.assertRaises(VisualConfigurationError):
            visual.generate_episode_visuals(
                series=_series(),
                episode=_episode(),
                output_dir=str(self.output),
                still_provider=_still_provider(enabled=False),
            )

    def test_shot_files_are_numbered_in_narrative_order(self):
        results = visual.generate_episode_visuals(
            series=_series(),
            episode=_episode(),
            output_dir=str(self.output),
            still_provider=_still_provider(),
            motion_provider=_motion_provider(),
            max_animated_shots=0,
        )
        self.assertTrue(results[0].still_path.endswith("shot-001.png"))
        self.assertTrue(results[1].still_path.endswith("shot-002.png"))


class TestReferenceImages(VisualTestCase):
    def test_generates_references_only_for_characters_that_lack_one(self):
        """重画定妆图会让老剧集和新剧集里的同一个角色对不上。"""
        series_store.save_series(
            _series(cast=[_character("berry"), _character("mango", False)])
        )
        stills = _still_provider()
        stills.generate_still.side_effect = lambda **kwargs: (
            Path(kwargs["output_path"]).write_bytes(b"png"),
            StillResult(path=kwargs["output_path"], provider="f", model="f"),
        )[1]

        generated = visual.generate_reference_images(
            series=series_store.load_series("fruit-court"),
            output_dir=str(self.output),
            still_provider=stills,
        )
        self.assertEqual(generated, ["mango"])
        self.assertEqual(stills.generate_still.call_count, 1)

    def test_registers_the_reference_against_the_series(self):
        series_store.save_series(_series(cast=[_character("berry", False)]))
        stills = _still_provider()
        stills.generate_still.side_effect = lambda **kwargs: (
            Path(kwargs["output_path"]).write_bytes(b"png"),
            StillResult(path=kwargs["output_path"], provider="f", model="f"),
        )[1]

        visual.generate_reference_images(
            series=series_store.load_series("fruit-court"),
            output_dir=str(self.output),
            still_provider=stills,
        )
        stored = series_store.load_series("fruit-court")
        self.assertTrue(stored.character("berry").has_reference())
        self.assertEqual(stored.missing_references(), [])

    def test_an_unconfigured_provider_is_rejected(self):
        with self.assertRaises(VisualConfigurationError):
            visual.generate_reference_images(
                series=_series(),
                output_dir=str(self.output),
                still_provider=_still_provider(enabled=False),
            )


if __name__ == "__main__":
    unittest.main()
