#!/usr/bin/env python3
"""Gallery-build edits: photos appear one per beat in fixed slots on a solid background,
spaced apart (never overlapping), then the canvas clears and the next pattern starts.

    python3 gallery.py --config config_cheetah.json

The config's "timeline" is a list of steps, one per beat (or "beats": n):
    {"image": "<key>", "slot": "<slot name>"}   add a photo to the canvas
    {"clear": true}                              back to the plain background
Photos keep their own shape (never cropped), fit inside their slot box, never move.
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw

from moodboard import ROOT, analyze_audio, beat_time, decode_small, font, grade, hex_rgb, load_source


def boxes_overlap(a, b, gap):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return not (ax + aw + gap <= bx or bx + bw + gap <= ax or ay + ah + gap <= by or by + bh + gap <= ay)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    cfg = json.load(open(ap.parse_args().config))
    fps = cfg["fps"]
    W, H = cfg["size"]
    outdir = os.path.join(ROOT, "out", cfg["name"])
    os.makedirs(outdir, exist_ok=True)
    slots = cfg["slots"]
    bg = np.empty((H, W, 3), np.float32)
    bg[:] = hex_rgb(cfg["background"])

    # ---- beat grid (absolute beat times, snapped to frames; frame 0 = sound 0.00 s)
    au = analyze_audio(cfg, None)
    if au["placeholder"]:
        sys.exit(f"song not found: {cfg['song']}")
    beats = au["beats"]
    if "start_beat_index" in cfg:
        start = cfg["start_beat_index"]
    else:
        start = min(range(au["phase"], len(beats), 4), key=lambda i: abs(beats[i] - cfg.get("song_start", 0)))

    def onset_near(t, win=0.15):
        """Strongest onset within +-win seconds of t (for a hit that sits off the beat grid)."""
        ot, ov = au["onset_t"], au["onset"]
        m = np.where(np.abs(ot - t) <= win)[0]
        return float(ot[m[np.argmax(ov[m])]])

    steps = cfg["timeline"]
    pos, shots = 0, []
    for st in steps:
        t = 0.0 if pos == 0 else beat_time(beats, start + pos)
        if "at_onset_near" in st:
            t = onset_near(st["at_onset_near"])
        if "at" in st:  # exact time in seconds (e.g. a hit measured from the waveform)
            t = float(st["at"])
        shots.append(dict(st, t=t, start_frame=int(round(t * fps))))
        pos += st.get("beats", 1)
    t_end = cfg.get("end_at") or beat_time(beats, start + pos)
    N = int(round(t_end * fps))
    for a, b in zip(shots, shots[1:] + [dict(start_frame=N)]):
        a["end_frame"] = b["start_frame"]

    # ---- build each canvas state once
    srcs, layers, plates, errs = {}, [], [], []
    gap = cfg.get("min_gap_px", 40)
    for i, sh in enumerate(shots):
        if sh.get("clear"):
            layers = []
        else:
            key, slot = sh["image"], slots[sh["slot"]]
            if key not in srcs:
                srcs[key] = load_source(cfg, key)[0]
            im = srcs[key]
            s = min(slot["w"] / im.width, slot["h"] / im.height)
            w, h = int(round(im.width * s)), int(round(im.height * s))
            box = (int(slot["cx"] - w / 2), int(slot["cy"] - h / 2), w, h)
            for k, b in layers:
                if boxes_overlap(box, b, gap):
                    errs.append(f"step {i + 1}: {key} would sit closer than {gap}px to {k}")
            if box[0] < 0 or box[1] < 0 or box[0] + w > W or box[1] + h > H:
                errs.append(f"step {i + 1}: {key} runs off the canvas")
            layers = layers + [(key, box)]
        canvas = bg.copy()
        for key, (x, y, w, h) in layers:
            canvas[y:y + h, x:x + w] = grade(np.asarray(srcs[key].resize((w, h), Image.LANCZOS)), cfg["grade"])
        plates.append(canvas)
    if errs:
        sys.exit("layout errors:\n  " + "\n  ".join(errs))
    used = [s["image"] for s in shots if not s.get("clear")]
    if len(set(used)) != len(used):
        sys.exit("a photo is used twice")

    grain = np.random.default_rng(cfg["grade"].get("grain_seed", 0)).normal(0, cfg["grade"]["grain"], (H, W, 1)).astype(np.float32)
    song = os.path.join(ROOT, cfg["song"])
    mp4 = os.path.join(outdir, f"{cfg['version']}.mp4")
    silent = os.path.join(outdir, f"{cfg['version']}_silent.mp4")
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
           "-r", str(fps), "-i", "-", "-i", song, "-filter_complex",
           f"[1:a]atrim=0:{N / fps:.4f},asetpts=N/SR/TB,afade=t=in:d=0.04,afade=t=out:st={N / fps - 0.04:.4f}:d=0.04[a]",
           "-map", "0:v", "-map", "[a]", "-c:v", "libx264", "-profile:v", "high", "-pix_fmt", "yuv420p",
           "-crf", "18", "-preset", "veryfast", "-r", str(fps), "-c:a", "aac", "-b:a", "192k",
           "-movflags", "+faststart", "-t", f"{N / fps:.4f}", mp4]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    frames8 = []
    for sh, pl in zip(shots, plates):
        fr = np.clip(pl + grain, 0, 255).astype(np.uint8)
        frames8.append(fr)
        b = fr.tobytes()
        for _ in range(sh["start_frame"], sh["end_frame"]):
            p.stdin.write(b)
    p.stdin.close()
    if p.wait():
        sys.exit("ffmpeg failed")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", mp4, "-an", "-c:v", "copy", "-movflags",
                    "+faststart", silent], check=True)

    # ---- QA on the rendered file
    F = decode_small(mp4)
    d = np.abs(np.diff(F, axis=0)).mean((1, 2, 3))
    offs = []
    for sh in shots[1:]:
        f = sh["start_frame"]
        hit = max(range(max(f - 3, 1), min(f + 4, len(F))), key=lambda i: d[i - 1])
        offs.append(hit - sh["t"] * fps)
    still = [d[f - 1] for sh in shots for f in range(sh["start_frame"] + 1, sh["end_frame"]) if f < len(F)]
    offs = np.abs(np.array(offs))
    rep = [f"tempo {au['tempo']:.1f} BPM, {len(shots)} beats-steps, {len(used)} photos, {N / fps:.2f} s",
           f"beat sync: mean |offset| {offs.mean():.2f} frames, worst {offs.max():.2f} -> {'PASS' if offs.max() <= 1 else 'FAIL'}",
           f"stillness: max held-frame diff {max(still):.3f} -> {'PASS' if max(still) < 1 else 'FAIL'}",
           f"layout: no photo overlaps or comes within {gap}px of another, none cropped, none repeated", ""]
    for i, sh in enumerate(shots):
        rep.append(f"{i + 1:2d}. {sh['t']:6.2f}s  " + ("(clear to background)" if sh.get("clear") else f"{sh['slot']:<13} {sh['image']}"))
    open(os.path.join(outdir, f"{cfg['version']}_qa.txt"), "w").write("\n".join(rep) + "\n")

    tw, th = 180, 320
    cols = 8
    rows = (len(frames8) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * (tw + 10) + 10, rows * (th + 28) + 10), (30, 26, 26))
    dr = ImageDraw.Draw(sheet)
    for i, (fr, sh) in enumerate(zip(frames8, shots)):
        x, y = 10 + (i % cols) * (tw + 10), 10 + (i // cols) * (th + 28)
        sheet.paste(Image.fromarray(fr).resize((tw, th)), (x, y))
        dr.text((x, y + th + 4), f"{i + 1}. {sh['t']:.2f}s " + ("clear" if sh.get("clear") else sh["slot"]),
                fill=(235, 225, 225), font=font(12))
    sheet.save(os.path.join(outdir, f"{cfg['version']}_sequence.jpg"), quality=88)
    print("\n".join(rep))


if __name__ == "__main__":
    main()
