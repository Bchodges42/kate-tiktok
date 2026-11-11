#!/usr/bin/env python3
"""Steady-beat edits: one photo per beat at a fixed tempo, in one of two looks.

    python3 steady.py --config config_peace_single.json

  style "single": one whole photo at a time, centred as big as it fits on a solid background.
  style "pile":   an overlapping collage. Every photo lands on top of the pile at the same area
                  (so a cover is never smaller than what it covers), with a thin border and a soft
                  shadow, cycling through anchor spots that keep the pile balanced on screen.

No song is needed: cuts sit on an exact beat grid (bpm). The main MP4 carries a plain kick-drum
beat guide so the timing can be heard; the _silent.mp4 is for posting with any TikTok sound at
that tempo. If "song" is given in the config, it is used instead of the beat guide.
"""
import argparse
import json
import math
import os
import subprocess
import sys
import wave

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from moodboard import ROOT, decode_small, font, grade, hex_rgb, load_source


def beat_guide(path, bpm, beats, sr=44100):
    """A plain kick on every beat and a soft tick on every off-beat (no melody, no samples)."""
    n = int(round(beats * 60 / bpm * sr))
    y = np.zeros(n, np.float32)
    t = np.arange(int(0.25 * sr)) / sr
    kick = np.sin(2 * np.pi * (45 + 90 * np.exp(-t * 35)) * t) * np.exp(-t * 14)
    tick = np.random.default_rng(1).normal(0, 1, int(0.03 * sr)) * np.exp(-np.arange(int(0.03 * sr)) / sr * 160) * 0.18
    for b in range(beats):
        for off, s, g in ((0.0, kick, 0.8), (0.5, tick, 1.0)):
            i = int(round((b + off) * 60 / bpm * sr))
            seg = s[: max(0, min(len(s), n - i))]
            y[i:i + len(seg)] += g * seg
    pcm = (np.clip(y, -1, 1) * 32000).astype(np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr); w.writeframes(pcm.tobytes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    cfg = json.load(open(ap.parse_args().config))
    W, H = cfg["size"]
    fps, bpm = cfg["fps"], cfg["bpm"]
    outdir = os.path.join(ROOT, "out", cfg["name"])
    os.makedirs(outdir, exist_ok=True)
    order = cfg["order"]
    kinds = {k: v.get("kind") for k, v in cfg["images"].items()}
    errs = []
    if len(set(order)) != len(order):
        errs.append("a photo is used twice")
    for a, b in zip(order, order[1:]):
        if kinds.get(a) and kinds.get(a) == kinds.get(b):
            errs.append(f"two {kinds[a]} photos back to back: {a}, {b}")

    # ---- beat grid: photo i lands on beat i; the last photo holds to the end of the bar
    spb = 60.0 / bpm
    total_beats = int(math.ceil(len(order) / 4.0) * 4) if cfg.get("end_on_bar", True) else len(order)
    starts = [int(round(i * spb * fps)) for i in range(len(order))]
    N = int(round(total_beats * spb * fps))
    ends = starts[1:] + [N]

    def make_bg(spec):
        """A solid colour ("#RRGGBB") or {"checker": [c1, c2], "size": px}."""
        b = np.empty((H, W, 3), np.float32)
        if isinstance(spec, dict) and "checker" in spec:
            c1, c2 = (np.array(hex_rgb(c), np.float32) for c in spec["checker"])
            sz = spec.get("size", 90)
            yy, xx = np.mgrid[0:H, 0:W]
            b[:] = np.where((((xx // sz) + (yy // sz)) % 2 == 0)[..., None], c1, c2)
        else:
            b[:] = hex_rgb(spec)
        return b
    bg = make_bg(cfg["background"])
    srcs = {k: load_source(cfg, k)[0] for k in order}
    lowres = [k for k in order if max(srcs[k].size) < 500]
    plates, boxes, cover = [], [], []
    m = cfg.get("margin_px", 40)

    if cfg["style"] == "single":
        bord = cfg.get("border_px", 0)
        bcol = np.array(hex_rgb(cfg.get("border", "#FFFFFF")), np.float32)
        cyc, every = cfg.get("bg_cycle"), cfg.get("bg_every", 4)  # e.g. a new background colour every bar
        for i, k in enumerate(order):
            im = srcs[k]
            s = min((W - 2 * m - 2 * bord) / im.width, (H - 2 * m - 2 * bord) / im.height)
            w, h = int(round(im.width * s)), int(round(im.height * s))
            fw, fh = w + 2 * bord, h + 2 * bord
            x, y = (W - fw) // 2, int(round((H - fh) * cfg.get("center_y", 0.5)))
            c = make_bg(cyc[(i // every) % len(cyc)]) if cyc else bg.copy()
            c[y:y + fh, x:x + fw] = bcol
            c[y + bord:y + bord + h, x + bord:x + bord + w] = grade(np.asarray(im.resize((w, h), Image.LANCZOS)), cfg["grade"])
            plates.append(c); boxes.append((x, y, fw, fh)); cover.append(fw * fh / (W * H))
    else:  # pile
        p = cfg["pile"]
        bord = p.get("border_px", 14)
        # one area for every photo, capped so even the widest/tallest photo fits on screen at it
        fit = min(min((W - 2 * m - 2 * bord) ** 2 * im.height / im.width, (H - 2 * m - 2 * bord) ** 2 * im.width / im.height)
                  for im in srcs.values())
        area = min(p["area_frac"] * W * H, fit)
        bl = p.get("border", "#FFFFFF")
        bcols = [np.array(hex_rgb(c), np.float32) for c in (bl if isinstance(bl, list) else [bl])]
        spots = p["spots"]
        canvas = bg.copy()
        for i, k in enumerate(order):
            im = srcs[k]
            r = im.width / im.height
            h = math.sqrt(area / r); w = h * r
            w = min(w, W - 2 * m - 2 * bord); h = w / r  # a wide photo never runs off the sides
            if h > H - 2 * m - 2 * bord:
                h = H - 2 * m - 2 * bord; w = h * r
            w, h = int(round(w)), int(round(h))
            cx, cy = spots[i % len(spots)]
            fw, fh = w + 2 * bord, h + 2 * bord
            x = int(min(max(cx - fw / 2, m), W - m - fw))
            y = int(min(max(cy - fh / 2, m), H - m - fh))
            # soft shadow (static), then the bordered photo
            sh = Image.new("L", (W, H), 0)
            ImageDraw.Draw(sh).rectangle([x + 6, y + 10, x + fw + 6, y + fh + 10], fill=int(255 * p.get("shadow", 0.35)))
            a = np.asarray(sh.filter(ImageFilter.GaussianBlur(14)), np.float32)[..., None] / 255
            canvas = canvas * (1 - a)
            canvas[y:y + fh, x:x + fw] = bcols[i % len(bcols)]
            canvas[y + bord:y + bord + h, x + bord:x + bord + w] = grade(np.asarray(im.resize((w, h), Image.LANCZOS)), cfg["grade"])
            plates.append(canvas.copy()); boxes.append((x, y, fw, fh))
            cov = np.zeros((H, W), bool)
            for bx, by, bw, bh in boxes:
                cov[by:by + bh, bx:bx + bw] = True
            cover.append(cov.mean())
        areas = [b[2] * b[3] for b in boxes]
        for i in range(1, len(areas)):
            if areas[i] < 0.97 * areas[i - 1]:
                errs.append(f"{order[i]} is smaller than the photo it covers")
    for k, (x, y, w, h) in zip(order, boxes):
        if x < 0 or y < 0 or x + w > W or y + h > H:
            errs.append(f"{k} runs off the screen")
    if errs:
        sys.exit("rule errors:\n  " + "\n  ".join(errs))

    # ---- render
    grain = np.random.default_rng(cfg["grade"].get("grain_seed", 0)).normal(0, cfg["grade"]["grain"], (H, W, 1)).astype(np.float32)
    mp4 = os.path.join(outdir, f"{cfg['version']}.mp4")
    silent = os.path.join(outdir, f"{cfg['version']}_silent.mp4")
    if cfg.get("song"):
        audio = os.path.join(ROOT, cfg["song"])
    else:
        audio = os.path.join(outdir, f"{cfg['version']}_beat.wav")
        beat_guide(audio, bpm, total_beats)
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(fps),
           "-i", "-", "-i", audio, "-map", "0:v", "-map", "1:a", "-c:v", "libx264", "-profile:v", "high",
           "-pix_fmt", "yuv420p", "-crf", "18", "-preset", "veryfast", "-c:a", "aac", "-b:a", "192k",
           "-movflags", "+faststart", "-t", f"{N / fps:.4f}", mp4]
    pr = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    frames8 = []
    for pl, s0, s1 in zip(plates, starts, ends):
        fr = np.clip(pl + grain, 0, 255).astype(np.uint8)
        frames8.append(fr)
        for _ in range(s0, s1):
            pr.stdin.write(fr.tobytes())
    pr.stdin.close()
    if pr.wait():
        sys.exit("ffmpeg failed")
    if not cfg.get("song"):
        os.remove(audio)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", mp4, "-an", "-c:v", "copy", "-movflags", "+faststart", silent], check=True)

    # ---- QA on the rendered file
    F = decode_small(mp4)
    d = np.abs(np.diff(F, axis=0)).mean((1, 2, 3))
    offs = []
    for i, f in enumerate(starts[1:], 1):
        hit = max(range(max(f - 3, 1), min(f + 4, len(F))), key=lambda j: d[j - 1])
        offs.append(abs(hit - i * spb * fps))
    still = [d[f - 1] for s0, s1 in zip(starts, ends) for f in range(s0 + 1, s1) if f < len(F)]
    rep = [f"{bpm} BPM steady, one photo per beat ({spb:.3f} s), {len(order)} photos, {total_beats} beats = {N / fps:.2f} s, ends on a bar line",
           f"beat sync: worst offset {max(offs):.2f} frames -> {'PASS' if max(offs) <= 1 else 'FAIL'}",
           f"stillness: max held-frame diff {max(still):.3f} -> {'PASS' if max(still) < 1 else 'FAIL'}",
           "photos: whole (no cropping), own shape (no stretching), none repeated -> PASS",
           "variety: no two photos of the same kind back to back -> PASS"]
    if cfg["style"] == "pile":
        rep.append("overlap: every photo lands at the same area as the one it covers, never smaller -> PASS")
        rep.append(f"screen filled: {cover[3]:.0%} after 4 photos, {cover[-1]:.0%} at the end")
    else:
        rep.append(f"screen filled per photo: {min(cover):.0%}-{max(cover):.0%} (whole photos; wide ones are shorter)")
    if lowres:
        rep.append("soft sources (under 500 px): " + ", ".join(lowres))
    rep.append("")
    for i, k in enumerate(order):
        rep.append(f"{i + 1:2d}. {starts[i] / fps:6.2f}s  {kinds.get(k, ''):<9} {k}")
    open(os.path.join(outdir, f"{cfg['version']}_qa.txt"), "w").write("\n".join(rep) + "\n")

    tw, th, cols = 180, 320, 8
    rows = (len(frames8) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * (tw + 10) + 10, rows * (th + 28) + 10), (30, 26, 26))
    dr = ImageDraw.Draw(sheet)
    for i, fr in enumerate(frames8):
        x, y = 10 + (i % cols) * (tw + 10), 10 + (i // cols) * (th + 28)
        sheet.paste(Image.fromarray(fr).resize((tw, th)), (x, y))
        dr.text((x, y + th + 4), f"{i + 1}. {starts[i] / fps:.2f}s {kinds.get(order[i], '')}", fill=(235, 225, 225), font=font(12))
    sheet.save(os.path.join(outdir, f"{cfg['version']}_sequence.jpg"), quality=88)
    print("\n".join(rep[:8]))


if __name__ == "__main__":
    main()
