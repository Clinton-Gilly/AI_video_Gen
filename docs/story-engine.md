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

## Visual backends

`app/services/visual/` turns the script into per-shot assets. Two protocols,
registered by name in `visual/__init__.py`, so adding a vendor is one line:

| Protocol | Default | Job |
|---|---|---|
| `StillProvider` | `gemini` | one image per shot, conditioned on cast reference stills |
| `MotionProvider` | `kling` | image-to-video, only for the shots the planner picks |

Gemini is the still default because `google-genai` is already a dependency, and
it reuses `gemini_api_key` from the LLM section — one account covers both.

### The cost control

Image-to-video bills per second and dominates the bill; stills are rounding
error. So the lever that matters is **how many shots you animate**, not which
model you buy.

`planner.plan_episode()` scores every shot and spends a fixed budget
(`max_animated_shots`, default 4) on the highest-value ones. Everything else
renders as a still with a slow Ken Burns move, which costs nothing.

Scoring favours what viewers actually notice: hook / turn / cliffhanger beats,
close-ups, shots with dialogue (lip movement is where stills give themselves
away), and handheld moves that Ken Burns physically cannot fake. Insert shots of
static objects score lowest — they are the cheapest thing to leave still.

At ~13 shots in a 45-second episode, animating 4 costs roughly a third of
animating all of them. `planner.estimate_cost()` reports the difference against
full motion so the budget is a number, not a guess. It takes prices as
arguments — vendor pricing moves too fast to hardcode.

Set `max_animated_shots = 0` to render stills only and spend nothing on video.

### Generation order

`generate_episode_visuals()` refuses to start when any character lacks a
reference still, and refuses when the plan needs motion but no motion provider
is configured — both checked *before* the first paid call, because discovering
either halfway through means paying twice.

## Assembly

`app/services/drama_assembly.py` turns shot visuals into a finished MP4. Three
things differ from stock-footage assembly:

**Audio decides shot length, not the script.** `Shot.duration` is a writing-time
estimate; the real length is however long the line takes to say. Cutting to the
script truncates dialogue, so every shot is stretched to its synthesized audio.
An animated clip that runs short is extended by freezing its last frame —
looping would show a visible jump back, worst of all on a talking shot.

**Every character speaks in their own voice.** `Character.voice_name` is read
here for the first time: dialogue uses the speaker's voice, narration uses the
series narrator, and a character with no voice set falls back to the narrator.
This is what makes a multi-character scene sound like a conversation instead of
one person reading a script.

**Captions are drama captions.** `CaptionStyle` drives size and position:
emphasis and title cards are larger and higher up; narration and dialogue sit in
the lower third, clear of faces. Emphasis captions are upper-cased, because that
is what the format looks like.

A single shot whose TTS fails degrades to a silent shot rather than failing the
episode — losing one line is much cheaper than losing the render.

## The Web UI

`webui/pages/2_Character_Drama.py` is a separate Streamlit page, deliberately
not part of `webui/Main.py`: that file is close to five thousand lines of
stock-footage controls, and the two pipelines share almost no widgets.

Three tabs follow the order of work: **Series** (create the bible and cast),
**Cast** (generate reference stills, assign a voice per character), **Episode**
(write Part N, choose the animation budget, watch the result). The sidebar
reports which backends are actually configured, and the episode tab can stop
after `script` so you can read what was written before spending anything on
images.

Run it with `webui.sh` / `webui.bat` as before, then pick **Character Drama**
from the page list.

## Background music

Drama episodes take music the same three ways stock videos do:

| `bgm_type` | Behaviour |
|---|---|
| `random` / a file | Track from the built-in library or your upload, looped to episode length and faded out. No API key. |
| `sonilo` / `elevenlabs` | The provider watches the finished episode and scores it. |

The generative providers need a video to watch, so drama assembles twice: once
into a draft with no music, then again with the generated track. That costs an
extra encode and is only paid when a generative provider is selected.

If a generative provider fails, the episode ships **silent rather than falling
back to the song library**, and the failure is recorded in the task's
`warnings`. A silent fallback would leave you believing you were listening to
music written for that episode.

## Connections page

`webui/pages/3_Connections.py` collects every provider key in one place and
tests it on demand — previously the only way to find out whether a key worked
was to run a job and have it fail partway through, after spending on the stages
that came first.

Every probe is free by construction:

| Provider | Probe |
|---|---|
| LLM (all of them — OpenAI, Qwen, Gemini, Ollama, Moonshot, DeepSeek…) | A two-word completion. The only check that covers key, base URL, model name and balance at once. |
| Gemini stills | Lists models. Generating an image would bill a request. |
| Kling motion | Queries a task id that cannot exist. A "not found" reply proves the key authenticated; submitting a real job bills per second. |
| Sonilo / ElevenLabs | Their free account endpoints. |
| Pexels / Pixabay | A one-item read query. |

Failures report the actual reason — key not set, key rejected, rate limited,
provider down, DNS failure — because "connection failed" tells you nothing about
what to do next. Rate limiting is deliberately not reported as a bad key: the
key works, it is just throttled.

The **Test everything** tab skips providers with no key rather than reporting
them as failures, so one real error is not lost in a wall of red.

## Not yet built

Still to come:

- Auto-publishing drama episodes through `upload_post`.
