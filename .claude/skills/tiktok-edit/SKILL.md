---
name: tiktok-edit
description: Make a beat-synced TikTok mood-board or fan edit from a Pinterest board (or a folder of photos), a song or TikTok sound, and a few design directions. Use whenever the user gives a Pinterest link, a song, a screen recording of a sound, or asks for an edit, a mood board video, a collage video, another version of an edit, or captions and hashtags for one.
---

# TikTok edit

Turn three things into a finished, checked, postable edit:

1. **A board**: a Pinterest board link (`pin.it/...` or `pinterest.com/<user>/<board>`), or photos the user uploads.
2. **A song**: an audio file, a screen recording of the TikTok sound, a song name, or nothing.
3. **Directions**: anything from "cream background, overlapping" to "you choose".

The hard rules live in `CLAUDE.md` at the repo root and always apply. This skill adds the working method, the account's taste and the engagement defaults. Read these before planning:

- `taste.md`: what this account has liked and rejected. Follow it unless the user says otherwise for this edit.
- `engagement.md`: length, hook, pacing and loop defaults, plus the post kit that ships with every edit.
- `styles.md`: which tool and starting config fit which look, and how each tool lines up with a song.

Do not ask questions the defaults already answer. Ask only when a hard rule would have to be broken, when a named person must be recognised (never identify anyone from a face; use captions, file names or the user's word), or when the user must supply something only they have (a song file, a secret board).

## 1. Read the brief

Write down, before touching photos:

| Item | From the user, or the default |
| --- | --- |
| Niche and search phrase | What a fan would type: "haymitch abernathy aesthetic", "coastal winter aesthetic". Used for the hook, caption and file names. |
| Vibe | One tight world: palette, era, mood. |
| Style | The look they asked for, or pick from `styles.md`. |
| Length | 8 to 15 s unless they ask for longer (see `engagement.md`). |
| Background, borders | Their hex or picture; otherwise a colour drawn from the board's own palette. Thin borders. |
| Anchors | Who or what goes first, what stays, what to leave out, any pattern to keep. |
| Versions | One main and one alternate that differs in one clear way, unless they ask for a set. |

## 2. Get the photos

```
python3 .claude/skills/tiktok-edit/scripts/fetch_board.py <link> --out <scratchpad>/board-<slug>
```

It saves every pin whole at the largest size, a `board.json` and numbered contact sheets. The feed returns at most the newest 50 pins; if the board has more, say which are missing. A secret board returns nothing: ask the user to make it public or upload the photos.

**Board sections.** The feed covers the whole board and does not say which section a pin is in. When the user names a section, open the section's public page (`pinterest.com/<user>/<board>/<section-slug>/`) in the pre-installed headless browser, scroll it, collect the pin image hashes it shows, and keep only the fetched pins whose image URL carries one of those hashes. The page needs the host `s.pinimg.com` allowed. Never guess a section from what the photos look like.

Then:

1. **Look at every contact sheet yourself.** Do not plan from file names or descriptions.
2. **Curate to one tight world.** Leave out off-palette photos, near-duplicates, soft sources (shorter side about 500 px or less), collages of other photos, screenshots with app chrome, and any photo that carries a creator's watermark or signature. Never crop a creator's mark out; skip the photo. A plain white print border or a site overlay may be cropped as a junk border, noted in the config.
3. **Copy keepers into `images/`** with a short board prefix and a descriptive name (`cw-snowy-lighthouse.jpg`). Keep the photo whole; only downsize files above 2400 px on the long side.
4. **Tag each keeper** in the config: `kind` (what it shows, so two of a kind never touch), shape, and anything the user's pattern needs.
5. **Keep a `not_used` note** in the config: which photos were left out and why, in a few words each.

If the user wants a photo that is not on the board (a named actor, a specific still), they must add it to the board or upload it. Do not search the web for images of people.

## 3. Map the song

| What the user gave | What to do |
| --- | --- |
| An audio file or screen recording | `ffmpeg -y -i <file> -vn -ac 2 -ar 44100 audio/<name>.wav`, then `python3 .claude/skills/tiktok-edit/scripts/song_map.py audio/<name>.wav --length 12`. Frame 0 of the video is sound 0.00 s, so never trim the start of a TikTok sound. |
| Only a song name | A song cannot be downloaded here. Ask once for a screen recording of the TikTok sound. If they would rather not, build on a steady grid at the song's known tempo and say the sync is to the tempo, not to the recording. |
| Nothing | Pick a tempo that suits the vibe (see `styles.md`) and build on a steady grid. The MP4 carries a plain beat guide; the silent copy is for a TikTok sound at that tempo. |

Read the map before planning: where the music ends, bar lines, the big hits, quiet intros, where a voice or melody takes over. Listen for what the user describes ("she goes up and down on the held note") and sync to that, not only to the drum grid. Confirm half or double time against the reference clip.

## 4. Plan the edit

Work in this order, and write the reasoning into the config's `_comment` fields:

1. **Length and count.** Pick the section of the song, then the number of photos that fits it at 0.2 to 0.6 s per photo. Curate down to that count; do not stretch the edit to fit the board.
2. **Hook.** The first frame is the most recognisable or striking photo for the niche, on screen at 0.00 s. The first 2 seconds are the busiest part of the edit.
3. **Order.** No two of a kind back to back, including around the loop. Alternate scale and subject. Put the strongest remaining photo on the biggest hit near the end.
4. **Loop.** End on a bar line, on a photo that leads back to the first. No intro, outro, fade to black or end card.
5. **Layout.** Search for the arrangement that makes photos biggest while obeying every rule in `CLAUDE.md`. Vary the arrangement between slides.
6. **Transitions.** Hard cuts on the beat by default; soft blends or in-place fades that finish on the beat when the user asks for smooth. Nothing slides, drops, zooms or wipes.

## 5. Render and check

One config per version, named `config_<board>_<style>.json`, rendered with the tool from `styles.md`:

```
python3 <tool>.py --config config_<board>_<style>.json
```

If no tool can do the look, extend a tool or add one. Keep it to one script plus one config per edit, and add a line for a new tool under "Tools in this repo" in `CLAUDE.md`.

Nothing goes to the user until all of this is true of the rendered file:

- Every item in the QA checklist in `CLAUDE.md` passes, using the tool's QA report.
- You have opened the sequence sheet and frames from the start, the middle, each transition type and the end, and fixed anything that looks awkward, soft, cramped or empty.
- The first frame works as a still: recognisable, whole, well filled.
- The last frame followed by the first reads as the next beat.

## 6. Publish the preview

```
.claude/skills/tiktok-edit/scripts/make_preview.sh out/<name>/<version>.mp4 <scratchpad>/prev
```

The shared viewing page is https://claude.ai/artifact/BMttNKuh7vvfpkbwuHYeKq. In a new session, read it first with the Artifact tool (`action: read`) and build on the saved file it names. Upload the 720p preview and the poster to that artifact as assets, then add a card at the top of the newest section using the same markup as the existing cards: a `<video>` with `poster` and `data-src` set to the two asset URLs, a status span `s-<first 8 of the video id>`, a Save button `d-<first 8>`, a title with the length, a meta line (photos, tempo, transition) and a one-paragraph note. New versions are added, not swapped in, unless the user says to replace one. Never send only a download link; the video must play in the page on a phone.

## 7. Save the work

Commit the configs, new photos and any tool changes, and push to the session's branch. Songs stay out of git (`audio/` is ignored). Do not open a pull request unless asked.

## 8. Report and post kit

Report in plain words, shortest first:

- The link to the viewing page and which card is new.
- What each version is: style, length, pace, number of photos, first and last photo.
- What the checks showed, including screen coverage and any soft source.
- Tradeoffs and guesses: photos left out and why, anything a rule cost, any choice the user did not make.

Then give the post kit from `engagement.md`: three caption options, two text-hook options for TikTok's text tool, 3 to 5 hashtags, the sound or tempo to post with, and the cover frame.
