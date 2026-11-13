#!/usr/bin/env python3
"""Map a song before planning cuts: where the music ends, tempo, beats, bar lines, big hits, and
the strongest short windows to build an edit on.

    python3 song_map.py audio/<song>.wav [--length 12] [--json out.json]

From a screen recording, extract the audio first:
    ffmpeg -y -i recording.mp4 -vn -ac 2 -ar 44100 audio/<name>.wav

Prints a plain summary and (with --json) every number, so cut times can be copied into a config.
The tempo can come out at half or double the felt speed; check it against the beat spacing by ear
or against the reference clip before trusting it.
"""
import argparse
import json

import librosa
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("song")
    ap.add_argument("--length", type=float, default=12.0, help="target edit length in seconds")
    ap.add_argument("--json")
    a = ap.parse_args()

    y, sr = librosa.load(a.song, sr=22050, mono=True)
    hop = 512
    dur = len(y) / sr
    rms = librosa.feature.rms(y=y, hop_length=hop)[0]
    t = librosa.times_like(rms, sr=sr, hop_length=hop)

    # Where the music ends: last moment louder than 8% of the loud level. Screen recordings run on
    # into silence, the control centre or TikTok's outro, which must not be synced to.
    loud = np.percentile(rms, 90)
    on = np.where(rms > 0.08 * loud)[0]
    music_end = float(t[on[-1]]) if len(on) else dur
    music_start = float(t[on[0]]) if len(on) else 0.0

    onset = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    tempo, bf = librosa.beat.beat_track(onset_envelope=onset, sr=sr, hop_length=hop, trim=False)
    tempo = float(np.atleast_1d(tempo)[0])
    beats = librosa.frames_to_time(bf, sr=sr, hop_length=hop)
    beats = beats[beats <= music_end + 0.05]

    # Bar lines: the beat phase (of 4) with the most low-end energy.
    mel = librosa.feature.melspectrogram(y=y, sr=sr, hop_length=hop, n_mels=64)
    low = mel[:12].sum(0)
    peak = np.array([low[b:b + 4].max() if b < len(low) else 0 for b in bf[:len(beats)]])
    phase = int(np.argmax([peak[p::4].mean() if len(peak[p::4]) else 0 for p in range(4)]))
    downbeats = beats[phase::4]

    # Big hits: the strongest onsets, at least 0.25 s apart.
    pk = librosa.util.peak_pick(onset, pre_max=6, post_max=6, pre_avg=20, post_avg=20, delta=0.5, wait=10)
    pk = [p for p in pk if t[min(p, len(t) - 1)] <= music_end]
    pk = sorted(pk, key=lambda p: -onset[p])[:24]
    hits = sorted((float(librosa.frames_to_time(p, sr=sr, hop_length=hop)), float(onset[p] / onset.max())) for p in pk)

    # Strongest windows: start on a bar line, last about --length, scored on loudness and activity.
    period = float(np.median(np.diff(beats))) if len(beats) > 1 else 60.0 / tempo
    nb = max(4, int(round(a.length / period / 4)) * 4)
    rn, onn = rms / (rms.max() + 1e-9), onset / (onset.max() + 1e-9)
    wins = []
    for i in range(phase, len(beats) - nb, 4):
        s, e = beats[i], beats[i + nb] if i + nb < len(beats) else beats[-1] + period
        if e > music_end + period:
            break
        ia, ib = np.searchsorted(t, s), np.searchsorted(t, e)
        wins.append((float(rn[ia:ib].mean() + 0.5 * onn[ia:ib].mean()), float(s), float(e)))
    wins = sorted(wins, reverse=True)[:3]

    # Loudness by second, to see intros, drops and quiet parts at a glance.
    per_sec = [float(rn[(t >= s) & (t < s + 1)].mean()) for s in range(int(music_end) + 1)]

    print(f"File length {dur:.2f} s. Music runs {music_start:.2f} to {music_end:.2f} s.")
    print(f"Tempo about {tempo:.1f} BPM (beat every {period:.3f} s = {period * 30:.1f} frames). "
          f"Half-beat {period / 2:.3f} s. Check for half/double time.")
    print(f"{len(beats)} beats; first at {beats[0]:.2f} s. Bar lines (every 4th beat): "
          + ", ".join(f"{d:.2f}" for d in downbeats[:16]) + (" ..." if len(downbeats) > 16 else ""))
    print(f"Strongest ~{nb * period:.1f} s windows ({nb} beats, starting on a bar line):")
    for sc, s, e in wins:
        print(f"  {s:.2f} to {e:.2f} s  (score {sc:.2f})")
    print("Biggest hits (time, strength 0-1): " + ", ".join(f"{h:.2f} ({v:.2f})" for h, v in hits))
    bar = "".join(" .:-=+*#%@"[min(9, int(v * 10))] for v in per_sec)
    print(f"Loudness by second: |{bar}|")

    if a.json:
        json.dump(dict(file=a.song, duration=dur, music_start=music_start, music_end=music_end, tempo=tempo,
                       beat_period=period, beats=[float(b) for b in beats], downbeats=[float(d) for d in downbeats],
                       hits=hits, windows=[dict(score=w[0], start=w[1], end=w[2]) for w in wins],
                       loudness_per_second=per_sec), open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
