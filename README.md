# kate-tiktok

Beat-synced aesthetic mood board edits for TikTok: one script, one config.

```
pip install librosa soundfile pillow numpy   # plus ffmpeg on PATH
python3 moodboard.py                          # renders every version in config.json
python3 moodboard.py --only C_pov_loop
```

## Next edit
1. Put the song in `audio/song.mp3` (or point `song` in `config.json` at it).
   Set `song_start` to the trending timestamp in seconds, or leave it `null` and the script picks
   the loudest 8–12 s window that starts on a downbeat.
2. Put images in `images/`, list the keepers under `images` (optional `crop` box in source
   pixels and `focus` point for the 9:16 crop), and record cuts under `rejected`.
3. Edit `sequence` (image + beats; 2 = hold) and `versions` (preset A or C, passes, beat_scale).

With no song file the script renders a `_PREVIEW-click` video on a placeholder grid at `fallback_bpm`.

## Outputs (`out/<name>/`)
- `<version>.mp4`: final with audio. `<version>_silent.mp4`: for adding the official sound in TikTok.
- `<version>_timeline.json`: image, start, end, dissolve frames per shot.
- `<version>_qa.txt`: stillness check, beat-sync offsets measured on the rendered file, loop gap.
- `<version>_sequence.jpg`, `<version>_transition_strip.jpg`, `curation_sheet.jpg`.

Rules the renderer enforces: images are pre-sized once and never move; the only change on screen is
an eased opacity dissolve (4–8 frames, longer into holds and on the loop) that finishes exactly on the
beat; cut times come from absolute beat times snapped to frames, so nothing drifts; grain is one
static layer.
