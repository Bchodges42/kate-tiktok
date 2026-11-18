#!/usr/bin/env python3
"""Pillow-pet "fill the space" edits: a featured photo sits dead centre and big photos land one by one
behind it, overlapping each other, until the screen is full; then the next featured photo starts a fresh slide.

    python3 pet_fill.py --config config_pp_fill_colour.json

Each slide ("pets" in the config) is: the featured photo alone, centred, held "pet_hold" steps; then one
backdrop photo per step; the finished slide holds "end_hold" steps; then the next slide starts straight away.

  * The featured photo is drawn last, on top, at its final size from its first frame. Backdrop photos are
    whole, in their own shape, never move once landed, and all share one area per slide (capped at the
    featured photo's own area), so every photo that lands on another is the same size or bigger.
  * "pattern" sets the order the ten spots fill: rows, rows_rev, spiral_cw, spiral_ccw. Same-kind photos
    never sit on neighbouring spots.
  * Cuts sit on an exact beat grid from 0.00 s at "bpm". "dissolve_frames": D makes each new photo fade in
    place over the D frames before its beat, so it is fully there on the beat; 0 = hard cuts.
No song: the MP4 carries a plain kick beat guide; the _silent.mp4 is for posting with a TikTok sound.
"""
import argparse
import json
import math
import os
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from moodboard import ROOT, decode_small, font, grade, hex_rgb, load_source
from steady import beat_guide

# ten centres (fractions of the screen), numbered 0-9; the featured photo covers the middle
SPOTS = [(0.28, 0.14), (0.72, 0.12), (0.78, 0.34), (0.22, 0.36), (0.78, 0.63), (0.22, 0.65), (0.72, 0.87), (0.28, 0.88), (0.50, 0.25), (0.50, 0.76)]
PATTERNS = {
    "rows": [1, 0, 8, 2, 3, 4, 5, 9, 6, 7],
    "rows_rev": [7, 6, 9, 5, 4, 3, 2, 8, 0, 1],
    "spiral_cw": [0, 8, 1, 2, 4, 6, 9, 7, 5, 3],
    "spiral_ccw": [1, 8, 0, 3, 5, 7, 9, 6, 4, 2],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    cfg = json.load(open(ap.parse_args().config))
    W, H = cfg["size"]
    fps, bpm = cfg["fps"], cfg["bpm"]
    spf = fps * 60 / bpm
    if abs(spf - round(spf)) > 1e-6:
        sys.exit(f"{bpm} BPM is {spf} frames per beat; pick a tempo that is a whole number of frames")
    spf = int(round(spf))
    D = cfg.get("dissolve_frames", 0)
    m, g, bd, pbd = cfg.get("margin_px", 36), cfg.get("gap_px", 18), cfg.get("border_px", 6), cfg.get("pet_border_px", 10)
    outdir = os.path.join(ROOT, "out", cfg["name"])
    os.makedirs(outdir, exist_ok=True)
    kinds = {k: v.get("kind") for k, v in cfg["images"].items()}
    errs, notes = [], []

    def bgfill(spec):
        b = np.empty((H, W, 3), np.float32)
        b[:] = hex_rgb(spec)
        return b

    def tile_area(key, area):
        im = load_source(cfg, key)[0]
        r = im.width / im.height
        ih = int(round(math.sqrt(area / r))) - 2 * bd
        iw = int(round(ih * r))
        t = np.empty((ih + 2 * bd, iw + 2 * bd, 3), np.float32)
        t[:] = hex_rgb(cfg.get("border", "#FFFFFF"))
        t[bd:bd + ih, bd:bd + iw] = grade(np.asarray(im.resize((iw, ih), Image.LANCZOS)), cfg["grade"])
        return t

    def pet_tile(key):
        im = load_source(cfg, key)[0]
        mw, mh = cfg.get("pet_max", [520, 640])
        s = min((mw - 2 * pbd) / im.width, (mh - 2 * pbd) / im.height)
        iw, ih = int(round(im.width * s)), int(round(im.height * s))
        t = np.empty((ih + 2 * pbd, iw + 2 * pbd, 3), np.float32)
        t[:] = hex_rgb(cfg.get("border", "#FFFFFF"))
        t[pbd:pbd + ih, pbd:pbd + iw] = grade(np.asarray(im.resize((iw, ih), Image.LANCZOS)), cfg["grade"])
        return t, im

    def shadow(w, h, x, y):
        a = Image.new("L", (W, H), 0)
        ImageDraw.Draw(a).rectangle([x + 4, y + 14, x + w - 4, y + h + 6], fill=int(255 * cfg.get("shadow", 0.30)))
        return np.asarray(a.filter(ImageFilter.GaussianBlur(24)), np.float32)[..., None] / 255

    slides = []   # one per pet
    for si, sl in enumerate(cfg["pets"]):
        bg = bgfill(sl.get("bg", cfg.get("background", "#F4ECDA")))
        pt, pim = pet_tile(sl["pet"])
        ph, pw = pt.shape[:2]
        px, py = (W - pw) // 2, (H - ph) // 2
        shade = shadow(pw, ph, px, py)
        spots = PATTERNS[sl["pattern"]]
        keys = sl["photos"]
        if len(keys) != len(spots):
            errs.append(f"slide {si + 1}: {len(keys)} photos for {len(spots)} spots")
            continue
        # one area for every backdrop photo on this slide, never above the featured photo's own area
        A = min(cfg.get("area_px", 430000), 0.97 * pw * ph)
        near = lambda a, b: math.hypot((SPOTS[a][0] - SPOTS[b][0]) * W, (SPOTS[a][1] - SPOTS[b][1]) * H) < cfg.get("near_px", 700)
        assign = {}

        def place(i, used):
            if i == len(spots):
                return True
            for k in keys:
                if k in used or any(near(spots[i], spots[j]) and kinds[assign[j]] == kinds[k] for j in range(i)):
                    continue
                assign[i] = k
                if place(i + 1, used | {k}):
                    return True
                del assign[i]
            return False
        if not place(0, set()):
            errs.append(f"slide {si + 1}: no arrangement keeps same-kind photos apart")
            continue
        layers = []
        for i, sp in enumerate(spots):
            tl = tile_area(assign[i], A)
            th, tw = tl.shape[:2]
            x = min(max(int(round(SPOTS[sp][0] * W - tw / 2)), m), W - m - tw)
            y = min(max(int(round(SPOTS[sp][1] * H - th / 2)), m), H - m - th)
            layers.append((tl, x, y))
            if tw > W - 2 * m or th > H - 2 * m:
                errs.append(f"{assign[i]} does not fit on the screen")
        if pw * ph < max(t.shape[0] * t.shape[1] for t, _, _ in layers):
            errs.append(f"{sl['pet']} is smaller than a photo it covers")
        slides.append(dict(bg=bg, pt=pt, px=px, py=py, shade=shade, layers=layers, order=[assign[i] for i in range(len(spots))],
                           pet=sl["pet"], name=sl.get("name", sl["pet"]), area=A))
    if errs:
        sys.exit("rule errors:\n  " + "\n  ".join(errs))
    def state(si, k):
        s = slides[si]
        c = s["bg"].copy()
        for t, x, y in s["layers"][:k]:
            c[y:y + t.shape[0], x:x + t.shape[1]] = t
        if cfg.get("shadow", 0.30):
            c *= 1 - s["shade"]
        c[s["py"]:s["py"] + s["pt"].shape[0], s["px"]:s["px"] + s["pt"].shape[1]] = s["pt"]
        return c

    # ---- timeline in steps: pet alone (pet_hold), then one photo per step, then the finished slide holds
    ph_, eh = cfg.get("pet_hold", 2), cfg.get("end_hold", 1)
    seq = []  # (start_step, slide, photos_visible)
    step = 0
    for si, s in enumerate(slides):
        n = len(s["layers"])
        seq.append((step, si, 0)); step += ph_
        for k in range(1, n + 1):
            seq.append((step, si, k)); step += 1
        step += eh
    bar = 4
    total = int(math.ceil(step / bar) * bar)
    N = total * spf
    starts = [st * spf for st, _, _ in seq]
    ends = starts[1:] + [N]

    grain = np.random.default_rng(cfg["grade"].get("grain_seed", 0)).normal(0, cfg["grade"]["grain"], (H, W, 1)).astype(np.float32)
    mp4 = os.path.join(outdir, f"{cfg['version']}.mp4")
    silent = os.path.join(outdir, f"{cfg['version']}_silent.mp4")
    audio = os.path.join(outdir, f"{cfg['version']}_beat.wav")
    beat_guide(audio, bpm, total)
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(fps), "-i", "-", "-i", audio,
           "-map", "0:v", "-map", "1:a", "-c:v", "libx264", "-profile:v", "high", "-pix_fmt", "yuv420p", "-crf", "18", "-preset", "veryfast",
           "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", "-t", f"{N / fps:.4f}", mp4]
    pr = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    thumbs = {}
    for n, (st, si, k) in enumerate(seq):
        cur = state(si, k)
        thumbs[(si, k)] = np.asarray(Image.fromarray(np.clip(cur, 0, 255).astype(np.uint8)).resize((180, 320), Image.LANCZOS))
        a_img = np.clip(cur + grain, 0, 255)
        # with "dissolve_frames", the next picture fades in place over the D frames before its beat,
        # so it is complete exactly on the beat and then holds perfectly still
        b_img = None
        if D and n + 1 < len(seq):
            b_img = np.clip(state(seq[n + 1][1], seq[n + 1][2]) + grain, 0, 255)
        s0, s1 = starts[n], ends[n]
        still_bytes = a_img.astype(np.uint8).tobytes()
        for f in range(s0, s1):
            jj = f - (s1 - D)
            if b_img is not None and jj >= 0:
                a = (jj + 1) / (D + 1)
                a = a * a * (3 - 2 * a)
                pr.stdin.write((a_img * (1 - a) + b_img * a + 0.5).astype(np.uint8).tobytes())
            else:
                pr.stdin.write(still_bytes)
    pr.stdin.close()
    if pr.wait():
        sys.exit("ffmpeg failed")
    os.remove(audio)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", mp4, "-an", "-c:v", "copy", "-movflags", "+faststart", silent], check=True)

    # ---- QA on the rendered file
    F = decode_small(mp4)
    d = np.abs(np.diff(F, axis=0)).mean((1, 2, 3))
    offs = []
    for n in range(1, len(seq)):
        f = starts[n]
        if D:  # the fade must be finished on the beat: the last frame that still changes is the beat itself
            win = range(max(f - D - 1, 1), min(f + 4, len(F)))
            peak = max(d[j - 1] for j in win)
            hit = max(j for j in win if d[j - 1] > 0.15 * peak)
        else:
            hit = max(range(max(f - 3, 1), min(f + 4, len(F))), key=lambda j: d[j - 1])
        offs.append(abs(hit - f))
    still = [d[f - 1] for n in range(len(seq)) for f in range(starts[n] + 1, ends[n] - (D if D and n + 1 < len(seq) else 0)) if f < len(F)]
    cover = []
    for si, s in enumerate(slides):
        mask = np.zeros((H, W), bool)
        for t, x, y in s["layers"]:
            mask[y:y + t.shape[0], x:x + t.shape[1]] = True
        mask[s["py"]:s["py"] + s["pt"].shape[0], s["px"]:s["px"] + s["pt"].shape[1]] = True
        cover.append(mask.mean())
    rep = [f"{bpm} BPM steady, {spf / fps:.3f} s per step; {len(slides)} slides of {len(slides[0]['layers'])} backdrop photos + the featured photo "
           f"(held {ph_} steps); {total} steps = {N / fps:.2f} s, ends on a bar line",
           (f"fade-in {D} frames ({D / fps:.2f} s) finishing on each beat; " if D else "hard cuts; ")
           + f"beat sync: worst offset {max(offs):.2f} frames -> {'PASS' if max(offs) <= 1 else 'FAIL'}",
           f"stillness: max held-frame diff {max(still):.3f} -> {'PASS' if max(still) < 1 else 'FAIL'}",
           "photos: whole, own shape, none repeated, nothing off screen -> PASS",
           "variety: no two same-kind photos on neighbouring spots -> PASS",
           f"screen filled when each slide is complete: {min(cover):.0%}-{max(cover):.0%}",
           "overlap: every backdrop photo has the same area on its slide ("
           + ", ".join(f"{s['name']} {s['area'] / 1000:.0f}k px" for s in slides) + "), and the featured photo is at least as big -> PASS"]
    rep.append("")
    for st, si, k in seq:
        if k == 0:
            rep.append(f"{st * spf / fps:6.2f}s  slide {si + 1}: {slides[si]['name']} alone, then " + ", ".join(slides[si]["order"]))
    open(os.path.join(outdir, f"{cfg['version']}_qa.txt"), "w").write("\n".join(rep) + "\n")
    tw, th = 180, 320
    cols = len(slides)
    rows = 3
    sheet = Image.new("RGB", (cols * (tw + 10) + 10, rows * (th + 28) + 10), (30, 26, 26))
    dr = ImageDraw.Draw(sheet)
    for si, s in enumerate(slides):
        nl = len(s["layers"])
        for r, k in enumerate((0, nl // 2, nl)):
            x, y = 10 + si * (tw + 10), 10 + r * (th + 28)
            sheet.paste(Image.fromarray(thumbs[(si, k)]), (x, y))
            dr.text((x, y + th + 4), f"{s['name']} {'alone' if k == 0 else f'+{k}'}", fill=(235, 225, 225), font=font(12))
    sheet.save(os.path.join(outdir, f"{cfg['version']}_sequence.jpg"), quality=88)
    print("\n".join(rep[:7]))


if __name__ == "__main__":
    main()
