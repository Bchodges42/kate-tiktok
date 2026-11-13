# Styles, tools and starting configs

Every tool takes `--config <file>` and writes the MP4, a `_silent.mp4`, a QA report and a sequence sheet into `out/<name>/`. Start a new edit by copying the closest config, renaming it `config_<board>_<style>.json`, and changing `name`, `version`, `images`, the order and the timing. Read the tool's docstring for the full list of fields.

## Pick the tool

| The user asks for | Tool and style | Start from |
| --- | --- | --- |
| One photo at a time, blending into the next | `steady.py`, `"style": "single"` with `dissolve_frames` and `dissolve_ease` | `config_coastal_winter_cream_dissolve.json` |
| One photo at a time in polaroid frames | `steady.py`, `single` with a `polaroid` block | `config_coastal_winter_cream_polaroid.json` |
| One photo at a time, hard cuts | `steady.py`, `single`, no `dissolve_frames` | `config_peace_single.json` |
| Overlapping, a centred tilted stack | `steady.py`, `"style": "pile"` with one spot and `tilt` | `config_coastal_winter_postcard_stack.json` |
| Overlapping, a scattered pile | `steady.py`, `pile` with several spots | `config_coastal_winter_navy_scatter.json` |
| Polaroids overlapping on a drawn page, soft fades | `scrapbook.py` with `drop_scale: 0`, `drop_ease: "smooth"`, `snow.count: 0`, no captions or title | `config_coastal_winter_blue_scrapbook.json` |
| Photos building a spaced arrangement, timed to a song's hits | `gallery.py` with `groups`, each with `at` times | `config_hunger_games.json` |
| The same on a steady beat, with named slots | `gallery.py` with `timeline` and `slots` | `config_cheetah_sunshine.json` |
| A stacked collage of swaps and covers on a slow song | `collage.py` | `config_christmas_song.json` |
| Full-screen photos, soft dissolves, 8 to 12 s | `moodboard.py` | `config.json` |
| A still slide of looping clips to drop into an edit by hand | `slide.py` | `config_slide_lit_grandmas_kitchen_dance.json` |

Avoid `"transition": "wipe"` in `steady.py` and the drop, caption, title and snow options in `scrapbook.py` unless the user asks for them; see `taste.md`.

## How each tool lines up with a song

Frame 0 of the video is always sound 0.00 s, because the silent copy is posted with the sound added in TikTok.

- **`gallery.py` with `groups`**: each photo has its own time in `at`, in seconds from the start of the audio. Take the times from `song_map.py` (beats, bar lines, hits). This is the tool for songs with pickups, tempo changes, brass or vocal hits, and slow intros.
- **`collage.py`**: finds musical events itself (harmony and melody changes on the song's pulse) and picks cuts from them. Set `audio.music_end`, `audio.video_end`, `audio.pulse_bpm` and `audio.min_gap`.
- **`moodboard.py`**: tracks beats in the song and chooses the strongest window, or the one nearest `song_start`.
- **`steady.py` and `scrapbook.py`**: cuts sit on an exact grid from 0.00 s at `bpm`. With `"song"` set, this only works when the song's tempo is steady and a beat falls on 0.00 s. Check `song_map.py`: the first beat within one frame of 0.00 and even beat spacing. If not, do not force it. Build the look in `gallery.py`, or extend the tool to take beat times.

## Steady-grid arithmetic

At 30 fps, a beat that is a whole number of frames never drifts:

| BPM | Frames per beat | Seconds per beat |
| --- | --- | --- |
| 60 | 30 | 1.0 |
| 75 | 24 | 0.8 |
| 90 | 20 | 0.667 |
| 100 | 18 | 0.6 |
| 120 | 15 | 0.5 |
| 150 | 12 | 0.4 |
| 180 | 10 | 0.333 |

Half-beat steps at 100 to 120 BPM give 0.25 to 0.3 s per photo, which suits fast, upbeat boards. One per beat at 120 to 150 BPM suits most others. For a calm, blended look use 75 to 100 BPM with one photo per beat and a blend of a third to a half of the beat.

Plan length from the count: photos times seconds per step, plus the final hold, rounded up to a bar line. Twenty photos at 120 BPM, one per beat, with a 2 beat final hold is 11 s.

## Background and border choices that have worked

| Board mood | Background | Border |
| --- | --- | --- |
| Warm, vintage, cosy | Cream `#F4ECDA` or off-white `#F7F4EE` | White, about 5 px |
| Coastal, wintry | Dusty blue `#93ABBE`, slate `#4F6478`, sand `#E3D8C4` | White or cream, 5 to 6 px |
| Bold fashion, animal print | Deep burgundy `#570B28` | None, with 16 to 20 px gaps |
| Couture, editorial | White `#FFFFFF` | None, with even 20 px gaps |
| Retro beach, bright | Warm sand `#F6E7C8` | White |

Pick the background from the board itself: sample the photos' common colours and choose a quiet one that lets them stand out. A background picture must already be 9:16; it is used whole.

## One grade for the whole edit

Keep `grade` light and the same for every photo: `warmth` 0 to 0.035, `lift` 4 to 14, `saturation` 0.96 to 1.0, `grain` 1.5 to 3.5 with a fixed `grain_seed`, so photos from different sources sit together and held frames stay perfectly still.
