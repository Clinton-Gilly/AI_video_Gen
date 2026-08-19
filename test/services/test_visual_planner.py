import unittest

from app.models.story import (
    CameraMove,
    Episode,
    Scene,
    Shot,
    ShotSize,
    StoryBeat,
)
from app.services.visual import planner


def _shot(index: int, **overrides) -> Shot:
    payload = {
        "index": index,
        "action": "reacts to the accusation",
        "character_ids": ["berry"],
        "narration": "The room went quiet.",
        "duration": 3.0,
    }
    payload.update(overrides)
    return Shot(**payload)


def _episode(scenes) -> Episode:
    return Episode(
        series_id="fruit-court",
        title="The Missing Receipt",
        scenes=scenes,
        cliffhanger="who signed it?",
    )


def _scene(index: int, beat: StoryBeat, shots) -> Scene:
    return Scene(index=index, beat=beat, setting="a courtroom", shots=shots)


class TestShotScoring(unittest.TestCase):
    def test_dialogue_outranks_silence_at_the_same_beat(self):
        """有台词的镜头需要口型，静帧最容易露馅。"""
        speaking = _shot(
            1, dialogue={"speaker_id": "berry", "line": "I kept every receipt."}
        )
        silent = _shot(1)
        self.assertGreater(
            planner.score_shot(speaking, StoryBeat.escalation),
            planner.score_shot(silent, StoryBeat.escalation),
        )

    def test_hook_outranks_setup(self):
        """钩子决定完播率，铺垫即使静止也不影响理解。"""
        shot = _shot(1)
        self.assertGreater(
            planner.score_shot(shot, StoryBeat.hook),
            planner.score_shot(shot, StoryBeat.setup),
        )

    def test_insert_shots_rank_lowest(self):
        """插入镜头拍静物，真实运动几乎看不出来，是最该省的地方。"""
        insert = _shot(1, shot_size=ShotSize.insert)
        close_up = _shot(1, shot_size=ShotSize.close_up)
        self.assertLess(
            planner.score_shot(insert, StoryBeat.escalation),
            planner.score_shot(close_up, StoryBeat.escalation),
        )

    def test_handheld_outranks_a_move_ken_burns_can_fake(self):
        """Ken Burns 只能做推拉和平移，手持抖动无法伪造。"""
        handheld = _shot(1, camera_move=CameraMove.handheld)
        push_in = _shot(1, camera_move=CameraMove.slow_push_in)
        self.assertGreater(
            planner.score_shot(handheld, StoryBeat.setup),
            planner.score_shot(push_in, StoryBeat.setup),
        )


class TestEpisodePlanning(unittest.TestCase):
    def _mixed_episode(self) -> Episode:
        return _episode(
            [
                _scene(
                    1,
                    StoryBeat.hook,
                    [
                        _shot(
                            1,
                            shot_size=ShotSize.close_up,
                            dialogue={"speaker_id": "berry", "line": "You lied."},
                        ),
                        _shot(2, shot_size=ShotSize.insert),
                    ],
                ),
                _scene(
                    2,
                    StoryBeat.setup,
                    [
                        _shot(1, shot_size=ShotSize.wide),
                        _shot(2, shot_size=ShotSize.insert),
                        _shot(3, shot_size=ShotSize.establishing),
                    ],
                ),
            ]
        )

    def test_respects_the_animation_budget(self):
        plans = planner.plan_episode(self._mixed_episode(), max_animated_shots=2)
        self.assertEqual(len(plans), 5)
        self.assertEqual(sum(1 for plan in plans if plan.animate), 2)

    def test_spends_the_budget_on_the_highest_value_shots(self):
        """预算应当花在钩子里的对话特写上，而不是铺垫里的插入镜头。"""
        plans = planner.plan_episode(self._mixed_episode(), max_animated_shots=1)
        animated = [plan for plan in plans if plan.animate]
        self.assertEqual(len(animated), 1)
        self.assertEqual(animated[0].beat, StoryBeat.hook)
        self.assertEqual(animated[0].shot.shot_size, ShotSize.close_up)

    def test_zero_budget_renders_stills_only(self):
        plans = planner.plan_episode(self._mixed_episode(), max_animated_shots=0)
        self.assertTrue(plans)
        self.assertFalse(any(plan.animate for plan in plans))

    def test_a_budget_larger_than_the_episode_animates_everything(self):
        plans = planner.plan_episode(self._mixed_episode(), max_animated_shots=99)
        self.assertTrue(all(plan.animate for plan in plans))

    def test_plans_are_returned_in_shot_order_not_score_order(self):
        """调用方按顺序逐镜执行，乱序会让镜头文件名和叙事顺序对不上。"""
        plans = planner.plan_episode(self._mixed_episode(), max_animated_shots=2)
        self.assertEqual([plan.shot.shot_size for plan in plans][0], ShotSize.close_up)
        self.assertEqual([plan.beat for plan in plans], [
            StoryBeat.hook,
            StoryBeat.hook,
            StoryBeat.setup,
            StoryBeat.setup,
            StoryBeat.setup,
        ])

    def test_planning_is_deterministic(self):
        """方案不稳定会让成本核算和重跑结果无法比较。"""
        episode = self._mixed_episode()
        first = planner.plan_episode(episode, max_animated_shots=2)
        second = planner.plan_episode(episode, max_animated_shots=2)
        self.assertEqual(
            [plan.animate for plan in first], [plan.animate for plan in second]
        )

    def test_negative_budget_is_rejected(self):
        with self.assertRaises(ValueError):
            planner.plan_episode(self._mixed_episode(), max_animated_shots=-1)


class TestCostEstimate(unittest.TestCase):
    def _plans(self, animated: int):
        episode = _episode(
            [
                _scene(
                    1,
                    StoryBeat.hook,
                    [_shot(index, duration=3.0) for index in range(1, 6)],
                )
            ]
        )
        return planner.plan_episode(episode, max_animated_shots=animated)

    def test_only_animated_seconds_are_billed_for_video(self):
        estimate = planner.estimate_cost(self._plans(2), cost_per_video_second=0.09)
        self.assertEqual(estimate["animated_shots"], 2)
        self.assertAlmostEqual(estimate["animated_seconds"], 6.0)
        self.assertAlmostEqual(estimate["video_cost"], 0.54)

    def test_reports_what_the_budget_saved_against_animating_everything(self):
        """省下的钱要是一个具体数字，否则无从判断预算设得对不对。"""
        estimate = planner.estimate_cost(self._plans(2), cost_per_video_second=0.09)
        # 5 shots x 3s = 15s at full motion, versus 6s under the budget.
        self.assertAlmostEqual(estimate["full_motion_cost"], 1.35)
        self.assertAlmostEqual(estimate["saved"], 0.81)

    def test_stills_are_billed_per_shot_including_unanimated_ones(self):
        estimate = planner.estimate_cost(
            self._plans(2), cost_per_video_second=0.09, cost_per_still=0.01
        )
        self.assertAlmostEqual(estimate["still_cost"], 0.05)
        self.assertAlmostEqual(estimate["total_cost"], 0.59)

    def test_negative_prices_are_rejected(self):
        with self.assertRaises(ValueError):
            planner.estimate_cost(self._plans(1), cost_per_video_second=-1.0)


if __name__ == "__main__":
    unittest.main()
