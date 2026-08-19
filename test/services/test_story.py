import json
import unittest
from unittest.mock import patch

from pydantic import ValidationError

from app.models.story import (
    CameraMove,
    Character,
    Episode,
    Scene,
    Series,
    Shot,
    ShotSize,
    StoryBeat,
)
from app.services import story


def _character(character_id: str = "berry", **overrides) -> Character:
    payload = {
        "id": character_id,
        "name": character_id.title(),
        "head_type": "strawberry",
        "appearance": "red hoodie, silver bracelet",
        "personality": "impatient",
    }
    payload.update(overrides)
    return Character(**payload)


def _series(**overrides) -> Series:
    payload = {
        "id": "fruit-court",
        "title": "Fruit Court",
        "premise": "Neighbours settle disputes in a tiny courtroom.",
        "moral_theme": "honesty",
        "visual_style": "soft studio lighting, muted pastel palette",
        "cast": [_character("berry"), _character("mango", head_type="mango")],
    }
    payload.update(overrides)
    return Series(**payload)


def _shot(index: int = 1, **overrides) -> Shot:
    payload = {
        "index": index,
        "action": "slams a folder onto the table",
        "character_ids": ["berry"],
        "narration": "She had been waiting three weeks for this.",
    }
    payload.update(overrides)
    return Shot(**payload)


def _episode_payload(**overrides) -> dict:
    payload = {
        "title": "The Missing Receipt",
        "hook": "She kept the receipt. He did not.",
        "moral": "Keep your promises in writing.",
        "cliffhanger": "Whose signature was on the second page?",
        "scenes": [
            {
                "beat": "hook",
                "setting": "a cramped courtroom, morning",
                "mood": "tense",
                "shots": [
                    {
                        "action": "slams a folder onto the table",
                        "character_ids": ["berry"],
                        "shot_size": "medium",
                        "camera_move": "slow_push_in",
                        "duration": 3.0,
                        "dialogue": {"speaker_id": "berry", "line": "I kept every receipt."},
                        "caption": "SHE KEPT EVERYTHING",
                        "caption_style": "emphasis",
                    },
                    {
                        "action": "stares at the folder without blinking",
                        "character_ids": ["mango"],
                        "shot_size": "close_up",
                        "duration": 2.5,
                        "narration": "He had not expected her to be prepared.",
                    },
                ],
            }
        ],
    }
    payload.update(overrides)
    return payload


class TestStoryModels(unittest.TestCase):
    def test_shot_without_any_spoken_or_written_content_is_rejected(self):
        """空镜头会在成片里变成静默留白，必须在结构校验阶段拦下。"""
        with self.assertRaises(ValidationError):
            Shot(index=1, action="the room is empty", character_ids=[])

    def test_speaker_must_be_present_in_the_shot(self):
        """说话人不在画面里意味着剧本自相矛盾，配音会对不上镜头。"""
        with self.assertRaises(ValidationError) as context:
            Shot(
                index=1,
                action="turns away",
                character_ids=["berry"],
                dialogue={"speaker_id": "mango", "line": "You are lying."},
            )
        self.assertIn("is not present in the shot", str(context.exception))

    def test_scene_rejects_shots_that_are_not_numbered_in_order(self):
        with self.assertRaises(ValidationError):
            Scene(
                index=1,
                beat=StoryBeat.setup,
                setting="a kitchen",
                shots=[_shot(1), _shot(3)],
            )

    def test_non_finale_episode_requires_a_cliffhanger(self):
        """连载靠悬念驱动下一集，缺悬念属于结构错误而不是风格问题。"""
        scene = Scene(
            index=1, beat=StoryBeat.hook, setting="a kitchen", shots=[_shot(1)]
        )
        with self.assertRaises(ValidationError) as context:
            Episode(
                series_id="fruit-court",
                title="Part One",
                scenes=[scene],
                cliffhanger="",
                is_finale=False,
            )
        self.assertIn("cliffhanger", str(context.exception))

    def test_finale_may_end_without_a_cliffhanger(self):
        scene = Scene(
            index=1, beat=StoryBeat.resolution, setting="a kitchen", shots=[_shot(1)]
        )
        episode = Episode(
            series_id="fruit-court",
            title="The End",
            scenes=[scene],
            cliffhanger="",
            is_finale=True,
        )
        self.assertTrue(episode.is_finale)

    def test_character_id_rejects_path_traversal(self):
        """角色 ID 会参与拼接落盘路径，必须排除目录逃逸。"""
        for unsafe in ["../escape", "cast/berry", "a b"]:
            with self.subTest(unsafe=unsafe), self.assertRaises(ValidationError):
                _character(unsafe)

    def test_series_rejects_duplicate_character_ids(self):
        with self.assertRaises(ValidationError):
            _series(cast=[_character("berry"), _character("berry")])

    def test_episode_reports_characters_in_first_appearance_order(self):
        scene = Scene(
            index=1,
            beat=StoryBeat.setup,
            setting="a kitchen",
            shots=[
                _shot(1, character_ids=["mango"]),
                _shot(2, character_ids=["berry", "mango"]),
            ],
        )
        episode = Episode(
            series_id="fruit-court",
            title="Order",
            scenes=[scene],
            cliffhanger="what next",
        )
        self.assertEqual(episode.character_ids(), ["mango", "berry"])

    def test_missing_references_lists_characters_without_a_reference_image(self):
        series = _series()
        series.cast[0].reference_image = "storage/series/fruit-court/cast/berry.png"
        self.assertEqual(series.missing_references(), ["mango"])


class TestEpisodeGeneration(unittest.TestCase):
    def _generate(self, response, **kwargs):
        with patch("app.services.llm._generate_response", return_value=response):
            return story.generate_episode(
                series=_series(), premise="a disputed receipt", **kwargs
            )

    def test_generates_an_episode_from_a_json_response(self):
        episode = self._generate(json.dumps(_episode_payload()))
        self.assertEqual(episode.part_number, 1)
        self.assertEqual(len(episode.shots()), 2)
        self.assertEqual(episode.duration(), 5.5)
        self.assertEqual(episode.cliffhanger, "Whose signature was on the second page?")

    def test_accepts_a_response_wrapped_in_a_markdown_fence(self):
        """非 OpenAI 供应商即使被要求返回裸 JSON 也常加 ```json 围栏。"""
        fenced = f"```json\n{json.dumps(_episode_payload())}\n```"
        self.assertEqual(len(self._generate(fenced).shots()), 2)

    def test_recovers_json_from_surrounding_commentary(self):
        noisy = f"Here is your episode:\n{json.dumps(_episode_payload())}\nEnjoy!"
        self.assertEqual(len(self._generate(noisy).shots()), 2)

    def test_renumbers_shots_the_model_numbered_wrongly(self):
        """镜头顺序由数组顺序决定，重编号比整集重来便宜。"""
        payload = _episode_payload()
        payload["scenes"][0]["shots"][0]["index"] = 7
        payload["scenes"][0]["shots"][1]["index"] = 7
        episode = self._generate(json.dumps(payload))
        self.assertEqual([shot.index for shot in episode.shots()], [1, 2])

    def test_normalizes_empty_dialogue_objects(self):
        """无台词镜头常返回空对象或 null，不应当作结构错误。"""
        payload = _episode_payload()
        payload["scenes"][0]["shots"][0]["dialogue"] = {}
        payload["scenes"][0]["shots"][0]["narration"] = "She said nothing at all."
        episode = self._generate(json.dumps(payload))
        self.assertIsNone(episode.shots()[0].dialogue)

    def test_rejects_an_episode_referencing_an_unknown_character(self):
        """未知角色拿不到定妆图，画面会逐镜漂移,因此在剧本阶段就要拦下。"""
        payload = _episode_payload()
        payload["scenes"][0]["shots"][1]["character_ids"] = ["durian"]
        with self.assertRaises(story.StoryGenerationError) as context:
            self._generate(json.dumps(payload))
        self.assertIn("durian", str(context.exception))

    def test_raises_after_exhausting_retries_on_unparsable_output(self):
        with patch(
            "app.services.llm._generate_response", return_value="not json at all"
        ) as generate:
            with self.assertRaises(story.StoryGenerationError):
                story.generate_episode(series=_series(), premise="a dispute")
        self.assertEqual(generate.call_count, story.MAX_STORY_RETRIES)

    def test_continuation_prompt_carries_the_previous_cliffhanger(self):
        prompt = story.build_episode_prompt(
            series=_series(),
            premise="the signature is examined",
            part_number=2,
            previous_cliffhanger="Whose signature was on the second page?",
        )
        self.assertIn("Whose signature was on the second page?", prompt)
        self.assertIn("Part 2", prompt)

    def test_opening_prompt_does_not_claim_a_previous_part(self):
        prompt = story.build_episode_prompt(
            series=_series(), premise="a dispute", part_number=1
        )
        self.assertIn("opening part", prompt)

    def test_finale_prompt_asks_for_an_empty_cliffhanger(self):
        prompt = story.build_episode_prompt(
            series=_series(), premise="a dispute", part_number=3, is_finale=True
        )
        self.assertIn("final part", prompt)


class TestSeriesGeneration(unittest.TestCase):
    def test_builds_a_series_from_a_json_response(self):
        response = json.dumps(
            {
                "title": "Fruit Court",
                "premise": "Neighbours argue in a tiny courtroom.",
                "moral_theme": "honesty",
                "visual_style": "soft studio lighting",
                "cast": [
                    {
                        "id": "berry",
                        "name": "Berry",
                        "head_type": "strawberry",
                        "appearance": "red hoodie",
                        "personality": "impatient",
                    }
                ],
            }
        )
        with patch("app.services.llm._generate_response", return_value=response):
            series = story.generate_series(
                concept="courtroom drama", series_id="fruit-court", cast_size=1
            )
        self.assertEqual(series.id, "fruit-court")
        self.assertEqual([member.id for member in series.cast], ["berry"])

    def test_custom_style_overrides_the_generated_style(self):
        """用户显式指定风格时不能被模型的自由发挥覆盖，否则跨集风格会漂移。"""
        response = json.dumps(
            {
                "title": "Fruit Court",
                "visual_style": "neon cyberpunk",
                "cast": [{"id": "berry", "name": "Berry"}],
            }
        )
        with patch("app.services.llm._generate_response", return_value=response):
            series = story.generate_series(
                concept="courtroom drama",
                series_id="fruit-court",
                cast_size=1,
                custom_style="flat pastel illustration",
            )
        self.assertEqual(series.visual_style, "flat pastel illustration")

    def test_rejects_a_series_with_no_characters(self):
        with patch(
            "app.services.llm._generate_response",
            return_value=json.dumps({"title": "Empty", "cast": []}),
        ):
            with self.assertRaises(story.StoryGenerationError):
                story.generate_series(concept="anything", series_id="empty")


class TestPromptAssembly(unittest.TestCase):
    def test_shot_prompt_injects_the_locked_appearance_not_the_shot_text(self):
        """角色长相取自系列档案而非镜头文本，这是跨镜头一致性的关键。"""
        series = _series()
        prompt = story.build_shot_image_prompt(
            series, _shot(1, character_ids=["berry"])
        )
        self.assertIn("red hoodie, silver bracelet", prompt)
        self.assertIn("a character with a strawberry for a head", prompt)
        self.assertIn("soft studio lighting", prompt)
        self.assertIn("slams a folder onto the table", prompt)

    def test_shot_prompt_suppresses_burned_in_text(self):
        """画面里不能带文字，字幕由渲染阶段统一压制,否则会出现双重字幕。"""
        prompt = story.build_shot_image_prompt(_series(), _shot())
        self.assertIn("no text", prompt)
        self.assertIn("no watermark", prompt)

    def test_shot_prompt_reflects_the_shot_size(self):
        prompt = story.build_shot_image_prompt(
            _series(), _shot(1, shot_size=ShotSize.extreme_close_up)
        )
        self.assertIn("extreme close-up", prompt)

    def test_shot_prompt_rejects_a_character_outside_the_cast(self):
        with self.assertRaises(ValueError):
            story.build_shot_image_prompt(
                _series(), _shot(1, character_ids=["durian"])
            )

    def test_reference_prompt_asks_for_a_neutral_reusable_pose(self):
        """定妆图的姿势、光线和背景会被后续镜头继承,越中性越好复用。"""
        prompt = story.build_reference_image_prompt(_series(), _character("berry"))
        self.assertIn("Neutral standing pose", prompt)
        self.assertIn("plain seamless mid-grey background", prompt)
        self.assertIn("soft studio lighting", prompt)

    def test_motion_prompt_describes_motion_without_restating_the_frame(self):
        """复述画面内容会诱使视频模型重新构图,破坏与静帧的一致性。"""
        shot = _shot(1, camera_move=CameraMove.slow_push_in)
        prompt = story.build_shot_motion_prompt(shot)
        self.assertIn("slow dolly push in", prompt)
        self.assertNotIn("slams a folder", prompt)

    def test_motion_prompt_requests_lip_movement_only_when_speaking(self):
        speaking = _shot(
            1, dialogue={"speaker_id": "berry", "line": "I kept every receipt."}
        )
        self.assertIn("lip and head movement", story.build_shot_motion_prompt(speaking))
        self.assertIn("ambient motion only", story.build_shot_motion_prompt(_shot()))


if __name__ == "__main__":
    unittest.main()
