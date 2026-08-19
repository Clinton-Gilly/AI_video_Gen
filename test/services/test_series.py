import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.models.story import Character, Episode, Scene, Series, Shot, StoryBeat
from app.services import series as series_store


def _series(series_id: str = "fruit-court", **overrides) -> Series:
    payload = {
        "id": series_id,
        "title": "Fruit Court",
        "premise": "Neighbours settle disputes in a tiny courtroom.",
        "moral_theme": "honesty",
        "visual_style": "soft studio lighting",
        "cast": [
            Character(id="berry", name="Berry", head_type="strawberry"),
            Character(id="mango", name="Mango", head_type="mango"),
        ],
    }
    payload.update(overrides)
    return Series(**payload)


def _episode(part_number: int = 1, cliffhanger: str = "who signed it?") -> Episode:
    shot = Shot(
        index=1,
        action="slams a folder onto the table",
        character_ids=["berry"],
        narration="She had waited three weeks.",
    )
    scene = Scene(
        index=1, beat=StoryBeat.hook, setting="a courtroom", shots=[shot]
    )
    return Episode(
        series_id="fruit-court",
        part_number=part_number,
        title=f"Part {part_number}",
        scenes=[scene],
        cliffhanger=cliffhanger,
        is_finale=not cliffhanger,
    )


class SeriesStoreTestCase(unittest.TestCase):
    """把 storage 根目录指向临时目录，避免用例之间互相污染。"""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.storage = Path(self.temp_dir.name)

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


class TestSeriesPersistence(SeriesStoreTestCase):
    def test_round_trips_a_series(self):
        saved = _series()
        series_store.save_series(saved)
        loaded = series_store.load_series("fruit-court")
        self.assertEqual(loaded.title, saved.title)
        self.assertEqual([m.id for m in loaded.cast], ["berry", "mango"])

    def test_loading_a_missing_series_raises(self):
        with self.assertRaises(series_store.SeriesStoreError):
            series_store.load_series("nothing-here")

    def test_rejects_identifiers_that_escape_the_storage_directory(self):
        """加载入口接收裸字符串，必须在拼接路径前独立校验。"""
        for unsafe in ["../../etc", "a/b", ""]:
            with self.subTest(unsafe=unsafe), self.assertRaises(ValueError):
                series_store.series_dir(unsafe)

    def test_lists_only_directories_holding_a_series_file(self):
        series_store.save_series(_series("fruit-court"))
        series_store.save_series(_series("veg-court"))
        (self.storage / "series" / "stray").mkdir(parents=True, exist_ok=True)
        self.assertEqual(series_store.list_series(), ["fruit-court", "veg-court"])

    def test_a_truncated_series_file_reports_a_store_error(self):
        """半个 JSON 应当报错而不是抛出裸的解析异常。"""
        series_store.save_series(_series())
        (self.storage / "series" / "fruit-court" / "series.json").write_text(
            '{"id": "fruit-court"', encoding="utf-8"
        )
        with self.assertRaises(series_store.SeriesStoreError):
            series_store.load_series("fruit-court")


class TestReferenceImages(SeriesStoreTestCase):
    def _write_image(self, name: str = "berry.png") -> Path:
        source = self.storage / name
        source.write_bytes(b"fake-png")
        return source

    def test_registering_a_reference_image_copies_it_into_the_series(self):
        """参考图必须复制：provider 的临时目录随时会被清掉。"""
        series_store.save_series(_series())
        source = self._write_image()

        character = series_store.register_reference_image(
            "fruit-court", "berry", str(source)
        )
        source.unlink()

        stored = self.storage / "series" / "fruit-court" / "cast" / "berry.png"
        self.assertTrue(stored.is_file())
        self.assertTrue(character.has_reference())
        self.assertEqual(
            series_store.load_series("fruit-court").character("berry").reference_image,
            character.reference_image,
        )

    def test_reference_path_is_stored_relative_so_storage_can_be_relocated(self):
        series_store.save_series(_series())
        character = series_store.register_reference_image(
            "fruit-court", "berry", str(self._write_image())
        )
        self.assertFalse(Path(character.reference_image).is_absolute())

    def test_a_registered_reference_resolves_back_to_a_readable_file(self):
        """
        写入和读取必须用同一个基准目录。基准不一致时登记会成功但解析不到，
        画面层只会当成"该角色没有定妆图"而静默退化，角色逐镜换脸且无报错。
        """
        series_store.save_series(_series())
        series_store.register_reference_image(
            "fruit-court", "berry", str(self._write_image())
        )

        series = series_store.load_series("fruit-court")
        resolved = series_store.reference_image_path(series, "berry")
        self.assertIsNotNone(resolved)
        self.assertTrue(resolved.is_file())
        self.assertEqual(resolved.read_bytes(), b"fake-png")

    def test_reference_path_is_none_for_a_character_without_one(self):
        series_store.save_series(_series())
        series = series_store.load_series("fruit-court")
        self.assertIsNone(series_store.reference_image_path(series, "mango"))

    def test_rejects_an_unsupported_image_type(self):
        series_store.save_series(_series())
        source = self.storage / "berry.txt"
        source.write_text("not an image", encoding="utf-8")
        with self.assertRaises(series_store.SeriesStoreError):
            series_store.register_reference_image("fruit-court", "berry", str(source))

    def test_rejects_a_character_outside_the_cast(self):
        series_store.save_series(_series())
        with self.assertRaises(series_store.SeriesStoreError):
            series_store.register_reference_image(
                "fruit-court", "durian", str(self._write_image())
            )

    def test_missing_references_drives_the_pre_render_check(self):
        series_store.save_series(_series())
        series_store.register_reference_image(
            "fruit-court", "berry", str(self._write_image())
        )
        series = series_store.load_series("fruit-court")
        self.assertEqual(series.missing_references(), ["mango"])


class TestEpisodeLinkage(SeriesStoreTestCase):
    def test_appending_an_episode_advances_the_series_count(self):
        series_store.save_series(_series())
        series_store.append_episode(_episode(1))
        series_store.append_episode(_episode(2))
        self.assertEqual(series_store.load_series("fruit-court").episode_count, 2)
        self.assertEqual(series_store.list_episodes("fruit-court"), [1, 2])

    def test_rewriting_an_earlier_episode_does_not_rewind_the_count(self):
        """重跑第 1 集不能让后续集号错位。"""
        series_store.save_series(_series())
        series_store.append_episode(_episode(1))
        series_store.append_episode(_episode(2))
        series_store.append_episode(_episode(1, cliffhanger="a revised hook"))
        self.assertEqual(series_store.load_series("fruit-court").episode_count, 2)

    def test_previous_cliffhanger_is_carried_into_the_next_part(self):
        series_store.save_series(_series())
        series_store.append_episode(_episode(1, cliffhanger="who signed it?"))
        self.assertEqual(
            series_store.previous_cliffhanger("fruit-court", 2), "who signed it?"
        )

    def test_first_part_has_no_previous_cliffhanger(self):
        series_store.save_series(_series())
        self.assertEqual(series_store.previous_cliffhanger("fruit-court", 1), "")

    def test_a_missing_previous_episode_degrades_to_an_opening(self):
        """缺前情不应当阻断生成，按开篇处理即可。"""
        series_store.save_series(_series())
        self.assertEqual(series_store.previous_cliffhanger("fruit-court", 5), "")

    def test_latest_episode_returns_the_highest_numbered_part(self):
        series_store.save_series(_series())
        series_store.append_episode(_episode(1))
        series_store.append_episode(_episode(2))
        self.assertEqual(series_store.latest_episode("fruit-court").part_number, 2)

    def test_list_episodes_ignores_unrelated_files(self):
        series_store.save_series(_series())
        series_store.append_episode(_episode(1))
        episodes_dir = self.storage / "series" / "fruit-court" / "episodes"
        (episodes_dir / "notes.txt").write_text("scratch", encoding="utf-8")
        (episodes_dir / "part-draft.json").write_text("{}", encoding="utf-8")
        self.assertEqual(series_store.list_episodes("fruit-court"), [1])

    def test_episode_files_are_zero_padded_for_natural_ordering(self):
        series_store.save_series(_series())
        series_store.append_episode(_episode(1))
        stored = self.storage / "series" / "fruit-court" / "episodes" / "part-001.json"
        self.assertTrue(stored.is_file())
        self.assertEqual(json.loads(stored.read_text(encoding="utf-8"))["part_number"], 1)


class TestHighLevelEntryPoints(SeriesStoreTestCase):
    def test_create_series_refuses_to_overwrite_an_existing_series(self):
        series_store.save_series(_series())
        with self.assertRaises(series_store.SeriesStoreError):
            series_store.create_series(concept="anything", series_id="fruit-court")

    def test_continue_series_generates_the_next_part_with_prior_context(self):
        series_store.save_series(_series())
        series_store.append_episode(_episode(1, cliffhanger="who signed it?"))

        with patch(
            "app.services.story.generate_episode", return_value=_episode(2)
        ) as generate:
            episode = series_store.continue_series("fruit-court", "the signature")

        self.assertEqual(episode.part_number, 2)
        kwargs = generate.call_args.kwargs
        self.assertEqual(kwargs["part_number"], 2)
        self.assertEqual(kwargs["previous_cliffhanger"], "who signed it?")
        self.assertEqual(series_store.load_series("fruit-court").episode_count, 2)


if __name__ == "__main__":
    unittest.main()
