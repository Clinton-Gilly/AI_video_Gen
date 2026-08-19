# Story engine

The story engine turns a one-line concept into a structured, character-consistent
mini-drama script. It is the front half of the character-drama pipeline: it
produces the *plan* for a video, plus the prompts the visual layer needs, but it
does not generate any images or video itself.

## Why it exists

The stock-footage pipeline (`llm.generate_script` → `llm.generate_terms` →
`material.download_videos`) writes one flat string of narration and then finds
unrelated B-roll to sit underneath it. Clip order does not matter there, and no
one appears twice.

A character mini-drama inverts every one of those assumptions:

- The same characters appear in dozens of shots and must not change appearance.
- Shots play in narrative order; shuffling them destroys the story.
- Different characters need different voices.
- Part 1 must set up a cliffhanger that Part 2 pays off, weeks later.

None of that fits into a string, so the engine models it explicitly.

## The model

```
Series                     stable across every episode
├── visual_style           the style bible, appended verbatim to every prompt
├── cast: [Character]      recurring cast
│   ├── appearance         locked description, reused in every shot
│   ├── reference_image    the character's locked reference still
│   └── voice_name         per-character TTS voice
└── Episode                one upload
    ├── hook               the first three seconds
    ├── cliffhanger        the question that forces the next part
    └── Scene              one story beat, one location
        └── Shot           one generated frame
            ├── action     what is visible — never appearance
            ├── shot_size  framing
            ├── camera_move  motion for the image-to-video stage
            ├── dialogue   speaker + line
            └── caption    on-screen emphasis text
```

The split between `Shot.action` and `Character.appearance` is the whole point.
Shot text describes only what happens; the appearance is injected from the cast
at prompt-assembly time so every shot receives a byte-identical description of
each character. Letting the model re-describe a character per shot is what makes
them drift.

## Consistency chain

For an image → animate backend, consistency comes from a chain of locked inputs:

1. `build_reference_image_prompt()` produces a neutral, flat-lit, plain-background
   reference still per character. Everything in that image gets inherited
   downstream, which is why the pose and lighting are deliberately boring.
2. `register_reference_image()` copies the approved still into the series
   directory, so it survives the provider's temp directory being cleaned up.
3. `build_shot_image_prompt()` assembles each shot's still prompt from the locked
   appearance + the style bible + the shot's own action and framing. The
   reference image is passed alongside it as the consistency condition.
4. `build_shot_motion_prompt()` describes only motion. It deliberately does not
   restate the frame contents — repeating them pushes the video model to
   recompose and breaks the match with the still.

## Validation

Structural errors are rejected before any generation credits are spent:

| Rejected | Why |
|---|---|
| A shot with no dialogue, narration, or caption | Renders as silent dead air |
| A speaker not present in their own shot | Voice will not match the frame |
| Non-sequential shot numbering | Story plays out of order |
| A non-finale episode with no cliffhanger | Nothing pulls viewers to Part 2 |
| A character not in the series cast | No reference image exists, so it drifts |
| More than 60 shots, or 12 scenes | Usually a runaway model, not a long story |

Shot numbering the model got wrong is renumbered from array order rather than
rejected — regenerating a whole episode over an index typo is not worth it.

## Storage

```
storage/series/<series_id>/
    series.json               bible + cast
    cast/<character_id>.png   reference stills
    episodes/part-001.json    per-episode scripts
```

Series data outlives task state on purpose: task directories get cleaned up, but
losing a reference image means the cast changes face mid-series. Writes go
through `utils.write_json_atomic()`, so an interrupted process cannot leave a
half-written file that makes the series unloadable.

Reference paths are stored relative to the storage root so the directory can be
moved or mounted into a container without breaking.

## Choosing a mode

The two production paths are selected with `VideoParams.mode`, which defaults to
`stock` so every existing caller is unaffected:

| Mode | Procedure |
|---|---|
| `stock` | LLM writes narration → keywords → stock clips from Pexels/Pixabay → TTS → subtitles → concat. Clip order does not matter. |
| `drama` | Story engine writes a structured episode → per-shot stills conditioned on the cast's reference images → animate each still → TTS → captions. Shot order is the story. |

From the command line:

```bash
# stock — unchanged, still the default
python cli.py --video-subject "How AI helps developers daily"

# drama — writes Part N of a series and stops at the script for review
python cli.py --mode drama \
  --series-id fruit-court \
  --episode-premise "A disputed receipt turns into a shouting match" \
  --stop-at script

# drama — final part, resolves instead of ending on a cliffhanger
python cli.py --mode drama --series-id fruit-court \
  --episode-premise "The signature is examined" --finale
```

`--series-id`, `--episode-premise` and `--finale` are rejected outside drama mode
rather than silently ignored, so a flag that cannot take effect fails loudly.

Drama mode currently runs to the end of the script stage and then stops with a
clear error, because no visual backend is wired in yet. It never falls back to
stock footage: a character drama backed by unrelated stock clips is a different
video, not a degraded one.

## Usage

```python
from app.services import series as series_store

# 1. Create the bible and cast (one LLM call).
series = series_store.create_series(
    concept="Neighbours settle petty disputes in a tiny fruit courtroom",
    series_id="fruit-court",
    cast_size=3,
    character_kind="fruit",
)

# 2. Generate reference stills, then register the approved ones.
#    (The visual provider supplies the image; the engine supplies the prompt.)
from app.services import story
prompt = story.build_reference_image_prompt(series, series.cast[0])
# ... generate image at /tmp/berry.png with your image backend ...
series_store.register_reference_image("fruit-court", "berry", "/tmp/berry.png")

# 3. Write Part 1.
episode = series_store.continue_series(
    "fruit-court", premise="A disputed receipt turns into a shouting match"
)

# 4. Write Part 2 — the previous cliffhanger is carried in automatically.
episode_two = series_store.continue_series(
    "fruit-court", premise="The signature on page two is examined"
)

# 5. Assemble prompts for the visual layer.
for shot in episode.shots():
    still_prompt = story.build_shot_image_prompt(series, shot)
    motion_prompt = story.build_shot_motion_prompt(shot)
```

Before rendering, check `series.missing_references()`. The engine deliberately
does not block script generation on missing reference images — writing a script
costs a fraction of what generating frames costs, so the check belongs at the
render entry point.

## Not yet built

The engine stops at prompts. Still to come:

- A visual provider interface (`generate_reference_image`, `generate_shot_still`,
  `animate_still`) with a concrete image → animate backend behind it.
- Per-character TTS routing — `Character.voice_name` is modelled but the audio
  stage still uses the single `VideoParams.voice_name`.
- A drama caption renderer for `CaptionStyle.emphasis` and `title_card`, distinct
  from the existing narration subtitles.
- A `video_source="story"` branch in `task.py` wiring this into the task pipeline
  alongside `pexels` and `loomloom`.
