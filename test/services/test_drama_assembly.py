import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from app.models.schema import VideoParams
from app.models.story import (
    CameraMove,
    CaptionStyle,
    Character,
    Episode,
    Scene,
    Series,
    Shot,
    ShotSize,
    StoryBeat,
)
from app.services import drama_assembly
from app.services.visual.base import ClipResult


def _series(**overrides) -> Series:
    payload = {
        "id": "fruit-court",
        "title": "Fruit Court",
        "visual_style": "soft studio lighting",
        "narrator_voice": "en-US-NarratorNeural",
        "cast": [
            Character(id="berry", name="Berry", voice_name="en-US-BerryNeural"),
            Character(id="mango", name="Mango"),
        ],
    }
    payload.update(overrides)
    return Series(**payload)


def _shot(index: int = 1, **overrides) -> Shot:
    payload = {
        "index": index,
        "action": "reacts",
        "character_ids": ["berry"],
        "narration": "The room went quiet.",
        "duration": 3.0,
    }
    payload.update(overrides)
    return Shot(**payload)


def _episode(shots) -> Episode:
    return Episode(
        series_id="fruit-court",
        title="The Missing Receipt",
        scenes=[
            Scene(index=1, beat=StoryBeat.hook, setting="a courtroom", shots=shots)
        ],
        cliffhanger="who signed it?",
    )


class TestVoiceRouting(unittest.TestCase):
    """每个角色用自己的音色，是多角色短剧听起来像对话的关键。"""

    def test_dialogue_uses_the_speakers_own_voice(self):
        shot = _shot(
            1, dialogue={"speaker_id": "berry", "line": "I kept it."}, narration=""
        )
        self.assertEqual(
            drama_assembly.resolve_voice(_series(), shot), "en-US-BerryNeural"
        )

    def test_a_speaker_without_a_voice_falls_back_to_the_narrator(self):
        shot = _shot(
            1,
            character_ids=["mango"],
            dialogue={"speaker_id": "mango", "line": "It wasn't me."},
            narration="",
        )
        self.assertEqual(
            drama_assembly.resolve_voice(_series(), shot), "en-US-NarratorNeural"
        )

    def test_narration_uses_the_narrator_voice_even_when_a_character_is_present(self):
        self.assertEqual(
            drama_assembly.resolve_voice(_series(), _shot()), "en-US-NarratorNeural"
        )

    def test_params_voice_is_the_last_resort(self):
        series = _series(narrator_voice="")
        self.assertEqual(
            drama_assembly.resolve_voice(series, _shot(), fallback="en-AU-Fallback"),
            "en-AU-Fallback",
        )


class TestShotAudio(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.params = VideoParams(video_subject="x")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_silent_shots_are_not_sent_to_tts(self):
        """只有强调字幕、没有念白的镜头不需要配音。"""
        episode = _episode([_shot(1, narration="", caption="SHE KNEW")])
        with patch("app.services.voice.tts") as tts:
            audio = drama_assembly.synthesize_shot_audio(
                _series(), episode, str(self.root), self.params
            )
        tts.assert_not_called()
        self.assertIsNone(audio[0].audio_path)
        self.assertGreaterEqual(audio[0].duration, drama_assembly.MIN_SILENT_SHOT_SECONDS)

    def test_a_failed_synthesis_degrades_that_shot_instead_of_the_episode(self):
        """一句话失败不该让整集作废。"""
        episode = _episode([_shot(1), _shot(2)])
        with patch("app.services.voice.tts", return_value=None):
            audio = drama_assembly.synthesize_shot_audio(
                _series(), episode, str(self.root), self.params
            )
        self.assertEqual(len(audio), 2)
        self.assertTrue(all(entry.audio_path is None for entry in audio))

    def test_each_shot_is_synthesized_with_its_resolved_voice(self):
        episode = _episode(
            [
                _shot(
                    1,
                    dialogue={"speaker_id": "berry", "line": "I kept it."},
                    narration="",
                ),
                _shot(2),
            ]
        )
        with patch("app.services.voice.tts", return_value=None) as tts:
            drama_assembly.synthesize_shot_audio(
                _series(), episode, str(self.root), self.params
            )
        used = [call.kwargs["voice_name"] for call in tts.call_args_list]
        self.assertEqual(used, ["en-US-BerryNeural", "en-US-NarratorNeural"])


class TestCaptionStyling(unittest.TestCase):
    def setUp(self):
        self.params = VideoParams(video_subject="x", font_size=48)

    def test_emphasis_captions_are_upper_cased(self):
        """这个格式的钩子文字几乎都是全大写，小写明显不像。"""
        shot = _shot(
            1, caption="she kept everything", caption_style=CaptionStyle.emphasis
        )
        with patch("app.services.video.wrap_text", return_value=("X", 10)) as wrap:
            drama_assembly.build_caption_clip(shot, 3.0, (1080, 1920), self.params)
        self.assertEqual(wrap.call_args.args[0], "SHE KEPT EVERYTHING")

    def test_dialogue_captions_keep_their_original_case(self):
        shot = _shot(
            1,
            caption="I kept every receipt.",
            caption_style=CaptionStyle.dialogue,
        )
        with patch("app.services.video.wrap_text", return_value=("X", 10)) as wrap:
            drama_assembly.build_caption_clip(shot, 3.0, (1080, 1920), self.params)
        self.assertEqual(wrap.call_args.args[0], "I kept every receipt.")

    def test_emphasis_and_title_cards_render_larger_than_narration(self):
        self.assertGreater(
            drama_assembly._CAPTION_SCALE[CaptionStyle.title_card],
            drama_assembly._CAPTION_SCALE[CaptionStyle.emphasis],
        )
        self.assertGreater(
            drama_assembly._CAPTION_SCALE[CaptionStyle.emphasis],
            drama_assembly._CAPTION_SCALE[CaptionStyle.narration],
        )

    def test_falls_back_to_the_spoken_text_when_no_caption_is_written(self):
        shot = _shot(1, caption="", narration="The room went quiet.")
        with patch("app.services.video.wrap_text", return_value=("X", 10)) as wrap:
            drama_assembly.build_caption_clip(shot, 3.0, (1080, 1920), self.params)
        self.assertEqual(wrap.call_args.args[0], "The room went quiet.")


class TestShotVisualFitting(unittest.TestCase):
    """镜头长度由配音决定，画面必须铺满它。"""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.still = self.root / "shot-001.png"
        Image.fromarray(
            np.full((320, 180, 3), 128, dtype=np.uint8)
        ).save(self.still)

    def tearDown(self):
        self.temp_dir.cleanup()

    def _result(self, clip_path=None) -> ClipResult:
        return ClipResult(
            shot_index=1,
            still_path=str(self.still),
            clip_path=clip_path,
            animated=clip_path is not None,
            duration=3.0,
        )

    def test_a_still_becomes_a_ken_burns_clip_of_the_audio_length(self):
        clip = drama_assembly.build_shot_visual(
            self._result(), _shot(1, camera_move=CameraMove.slow_push_in), 4.2,
            (180, 320),
        )
        self.assertAlmostEqual(clip.duration, 4.2, places=2)

    def test_a_static_shot_still_fills_the_audio_length(self):
        clip = drama_assembly.build_shot_visual(
            self._result(), _shot(1, camera_move=CameraMove.static), 2.5, (180, 320)
        )
        self.assertAlmostEqual(clip.duration, 2.5, places=2)

    def test_a_missing_clip_file_falls_back_to_the_still(self):
        """产物丢失不该让整集崩掉，静帧本身仍然可用。"""
        clip = drama_assembly.build_shot_visual(
            self._result(clip_path=str(self.root / "gone.mp4")),
            _shot(1),
            3.0,
            (180, 320),
        )
        self.assertAlmostEqual(clip.duration, 3.0, places=2)


class TestBackgroundMusic(unittest.TestCase):
    def setUp(self):
        self.params = VideoParams(video_subject="x", bgm_type="random", bgm_volume=0.2)

    def test_a_provider_generated_track_is_used_verbatim(self):
        """供应商按成片时长生成的配乐已经等长，不该再走曲库解析。"""
        with patch("app.services.video.get_bgm_file") as library:
            resolved = drama_assembly.resolve_bgm_file(
                self.params, bgm_file_override="/tmp/generated.mp3"
            )
        library.assert_not_called()
        self.assertEqual(resolved, "/tmp/generated.mp3")

    def test_an_empty_override_means_no_music_at_all(self):
        """
        选了视频转音乐供应商但生成失败时，不能让曲库悄悄顶上——用户会以为
        听到的是为这一集生成的配乐。
        """
        with patch("app.services.video.get_bgm_file") as library:
            resolved = drama_assembly.resolve_bgm_file(self.params, bgm_file_override="")
        library.assert_not_called()
        self.assertEqual(resolved, "")

    def test_no_override_falls_back_to_the_song_library(self):
        with patch(
            "app.services.video.get_bgm_file", return_value="/songs/a.mp3"
        ) as library:
            resolved = drama_assembly.resolve_bgm_file(self.params, bgm_file_override=None)
        library.assert_called_once()
        self.assertEqual(resolved, "/songs/a.mp3")

    def test_zero_volume_disables_music(self):
        params = VideoParams(video_subject="x", bgm_type="random", bgm_volume=0.0)
        with patch("app.services.video.get_bgm_file") as library:
            self.assertEqual(drama_assembly.resolve_bgm_file(params, None), "")
        library.assert_not_called()


class TestAssembleEpisode(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def _still(self, name: str) -> str:
        path = self.root / name
        Image.fromarray(np.full((320, 180, 3), 90, dtype=np.uint8)).save(path)
        return str(path)

    def test_rejects_a_mismatch_between_shots_and_visuals(self):
        """镜头和画面对不上说明上游出了问题，硬拼会把台词配到错的画面上。"""
        episode = _episode([_shot(1), _shot(2)])
        with self.assertRaises(drama_assembly.DramaAssemblyError):
            drama_assembly.assemble_episode(
                series=_series(),
                episode=episode,
                clips=[
                    ClipResult(1, self._still("a.png"), None, False, 3.0)
                ],
                output_dir=str(self.root / "out"),
                params=VideoParams(video_subject="x"),
            )

    def test_rejects_an_empty_visual_list(self):
        with self.assertRaises(drama_assembly.DramaAssemblyError):
            drama_assembly.assemble_episode(
                series=_series(),
                episode=_episode([_shot(1)]),
                clips=[],
                output_dir=str(self.root / "out"),
                params=VideoParams(video_subject="x"),
            )

    def test_renders_a_playable_file_end_to_end(self):
        """
        真正跑一次 MoviePy，确认镜头、字幕和拼接能产出可播放的文件。
        TTS 被打桩成无声，避免测试依赖网络和第三方配额。
        """
        episode = _episode(
            [
                _shot(1, caption="SHE KEPT IT", caption_style=CaptionStyle.emphasis),
                _shot(2, shot_size=ShotSize.insert, narration="Page two was blank."),
            ]
        )
        clips = [
            ClipResult(1, self._still("s1.png"), None, False, 2.0),
            ClipResult(2, self._still("s2.png"), None, False, 2.0),
        ]
        params = VideoParams(video_subject="x", video_aspect="9:16", font_size=40)

        with patch("app.services.voice.tts", return_value=None):
            output = drama_assembly.assemble_episode(
                series=_series(),
                episode=episode,
                clips=clips,
                output_dir=str(self.root / "out"),
                params=params,
            )

        self.assertTrue(os.path.exists(output))
        self.assertGreater(os.path.getsize(output), 0)
        self.assertTrue(output.endswith("episode-001.mp4"))


if __name__ == "__main__":
    unittest.main()
