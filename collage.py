#!/usr/bin/env python3
"""Stacked-collage mood board edits (cream canvas, hard cuts, overlapping photos).

    python3 collage.py --config config_christmas_song.json

Each photo keeps its own shape and sits on a cream canvas. On every cut a new
photo either replaces the canvas (swap), is re-cut bigger (pop), or lands on top
of the previous photo, centered (inset / cover). Photos are never cropped and never
move while on screen. Cuts land on musical events (pulse points where the
harmony changes), which suits slow songs with no drum beat.
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw

from moodboard import ROOT, cover, decode_small, font, grade, hex_rgb, load_source

OVERLAYS = {"inset", "cover"}


# --------------------------------------------------------------------------- audio events
def music_events(path, cfg):
    """Cut candidates: harmony/melody changes snapped to the song's steady sub-pulse."""
    import librosa
    a = cfg["audio"]
    y, sr = librosa.load(path, sr=22050, duration=a["music_end"])
    hop = 256
    on = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    tempo, grid = librosa.beat.beat_track(onset_envelope=on, sr=sr, hop_length=hop,
                                          start_bpm=a.get("pulse_bpm", 158), tightness=300,
                                          units="time", trim=False)
    C = librosa.feature.chroma_cqt(y=librosa.effects.harmonic(y), sr=sr, hop_length=hop)
    C = librosa.util.normalize(librosa.decompose.nn_filter(C, aggregate=np.median, metric="cosine"), axis=0)
    k = int(0.35 * sr / hop)
    nov = np.array([np.linalg.norm(C[:, i - k:i].mean(1) - C[:, i:i + k].mean(1)) if k < i < C.shape[1] - k else 0
                    for i in range(C.shape[1])])
    nov = nov / nov.max()
    pk = librosa.util.peak_pick(nov, pre_max=12, post_max=12, pre_avg=30, post_avg=30, delta=0.05, wait=25)
    t = librosa.times_like(nov, sr=sr, hop_length=hop)
    events = []
    for p in pk:
        g = grid[np.argmin(np.abs(grid - t[p]))]
        tt = g if abs(g - t[p]) <= a.get("snap", 0.13) else t[p]  # on the pulse when the pulse is there
        events.append((float(tt), float(nov[p])))
    return float(np.atleast_1d(tempo)[0]), grid, events


def pick_cuts(events, n_cuts, min_gap, end):
    """Keep the strongest events, n_cuts of them, at least min_gap apart (frame 0 is always a cut)."""
    ev = sorted([e for e in events if min_gap <= e[0] <= end - min_gap], key=lambda e: -e[1])
    chosen = [0.0]
    for t, s in ev:
        if all(abs(t - c) >= min_gap for c in chosen):
            chosen.append(t)
        if len(chosen) == n_cuts + 1:
            break
    if len(chosen) < n_cuts + 1:
        sys.exit(f"only {len(chosen) - 1} usable musical events for {n_cuts} cuts: lower min_gap or use fewer shots")
    return sorted(chosen)


# --------------------------------------------------------------------------- layout
def fit(im, box_w, box_h):
    """Largest size at the image's exact aspect ratio that fits the box (never cropped)."""
    s = min(box_w / im.width, box_h / im.height)
    return int(round(im.width * s)), int(round(im.height * s))


def place(cfg, shots):
    """Turn the shot list into per-shot layer stacks. Every photo is centered and shown whole.

    swap  - clear the canvas, photo at ~70% of its full size if a pop follows, else full size
    pop   - the same photo cut back in at full size
    inset - a smaller photo centered on top of the current one
    cover - a near-full-size photo centered on top; the one underneath peeks out around it
    'Full size' = as large as the photo fits in the max box (tall photos fill the height,
    wide photos fill the width).
    """
    W, H = cfg["size"]
    lay = cfg["layout"]
    cx, cy = W // 2, int(H * lay.get("center_y", 0.46))
    max_w, max_h = W * lay.get("max_width", 0.92), H * lay.get("max_height", 0.72)
    srcs, states, layers = {}, [], []

    def centered(w, h):
        return (cx - w // 2, cy - h // 2, w, h)

    for i, sh in enumerate(shots):
        typ, key = sh["type"], sh["image"]
        if key not in srcs:
            srcs[key] = load_source(cfg, key)[0]
        im = srcs[key]
        if typ == "swap":
            f = lay.get("pre_pop", 0.7) if i + 1 < len(shots) and shots[i + 1]["type"] == "pop" else 1.0
            layers = [(key, centered(*fit(im, max_w * f, max_h * f)))]
        elif typ == "pop":
            k0 = layers[-1][0]
            layers = [(k0, centered(*fit(srcs[k0], max_w, max_h)))]
        elif typ == "inset":
            _, (_, _, pw, ph) = layers[-1]
            s = lay.get("inset_scale", 0.68)
            layers = layers + [(key, centered(*fit(im, pw * s, ph * s)))]
        elif typ == "cover":
            s = lay.get("cover_scale", 0.88)
            layers = layers + [(key, centered(*fit(im, max_w * s, max_h * s)))]
        else:
            sys.exit(f"shot {i}: unknown transition type {typ!r} (swap, pop, inset, cover)")
        if len(layers) > 3:
            sys.exit(f"shot {i}: more than 3 stacked photos; put a swap before it")
        states.append(list(layers))
    return states, srcs


def plate_for(cfg, state, srcs, cache):
    """Render one still composition (all layers) once; reused for every frame of the shot."""
    W, H = cfg["size"]
    canvas = np.empty((H, W, 3), np.float32)
    canvas[:] = hex_rgb(cfg["layout"]["background"])
    for key, (x, y, w, h) in state:
        ck = (key, w, h)
        if ck not in cache:
            meta = cfg["images"].get(key, {})
            cache[ck] = grade(np.asarray(srcs[key].resize((w, h), Image.LANCZOS)), cfg["grade"])  # whole photo, no crop
        img = cache[ck]
        x0, y0 = max(x, 0), max(y, 0)
        x1, y1 = min(x + w, W), min(y + h, H)
        canvas[y0:y1, x0:x1] = img[y0 - y:y1 - y, x0 - x:x1 - x]
    return canvas


# --------------------------------------------------------------------------- checks on the plan
def check_plan(cfg, shots):
    errs = []
    tree = {k for k, m in cfg["images"].items() if m.get("tree")}
    new = [s for s in shots if s["type"] != "pop"]
    for a, b in zip(new, new[1:] + new[:1]):  # includes the loop back to image 1
        if (a["image"] in tree) == (b["image"] in tree):
            errs.append(f"tree/no-tree not alternating: {a['image']} -> {b['image']}")
    for a, b in zip(shots, shots[1:]):
        if a["type"] == b["type"]:
            errs.append(f"same transition twice in a row: {a['type']} ({a['image']} -> {b['image']})")
    if cfg.get("no_repeat_images", True):
        imgs = [s["image"] for s in new]
        if len(set(imgs)) != len(imgs):
            errs.append("an image is used twice")
    return errs


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = json.load(open(args.config))
    fps = cfg["fps"]
    W, H = cfg["size"]
    outdir = os.path.join(ROOT, "out", cfg["name"])
    os.makedirs(outdir, exist_ok=True)
    shots = cfg["shots"]
    errs = check_plan(cfg, shots)
    if errs:
        sys.exit("plan errors:\n  " + "\n  ".join(errs))

    song = os.path.join(ROOT, cfg["song"])
    a = cfg["audio"]
    tempo, grid, events = music_events(song, cfg)
    cuts = pick_cuts(events, len(shots) - 1, a["min_gap"], a["video_end"])
    ev_strength = {round(t, 3): v for t, v in events}
    N = int(round(a["video_end"] * fps))
    frames = [int(round(t * fps)) for t in cuts] + [N]
    for sh, t, f0, f1 in zip(shots, cuts, frames, frames[1:]):
        sh.update(t=round(t, 3), start_frame=f0, end_frame=f1)

    states, srcs = place(cfg, shots)
    cache = {}
    grain = np.random.default_rng(cfg["grade"].get("grain_seed", 0)).normal(0, cfg["grade"]["grain"], (H, W, 1)).astype(np.float32)
    mp4 = os.path.join(outdir, f"{cfg['version']}.mp4")
    silent = os.path.join(outdir, f"{cfg['version']}_silent.mp4")
    fade = 0.04
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
           "-r", str(fps), "-i", "-", "-i", song, "-filter_complex",
           f"[1:a]atrim=0:{N / fps:.4f},asetpts=N/SR/TB,afade=t=in:d={fade},afade=t=out:st={N / fps - 0.15:.4f}:d=0.15[a]",
           "-map", "0:v", "-map", "[a]", "-c:v", "libx264", "-profile:v", "high", "-pix_fmt", "yuv420p",
           "-crf", "18", "-preset", "veryfast", "-r", str(fps), "-c:a", "aac", "-b:a", "192k",
           "-movflags", "+faststart", "-t", f"{N / fps:.4f}", mp4]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    plates = []
    for sh, st in zip(shots, states):
        fr = np.clip(plate_for(cfg, st, srcs, cache) + grain, 0, 255).astype(np.uint8)
        plates.append(fr)
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
    rep = [f"pulse {tempo:.1f} BPM (half-beats of a ~{tempo / 2:.0f} BPM ballad), {len(events)} musical events, "
           f"{len(shots)} shots / {len({s['image'] for s in shots})} images, {N / fps:.2f} s"]
    offs, still = [], []
    for sh in shots[1:]:
        f = sh["start_frame"]
        win = range(max(f - 3, 1), min(f + 4, len(F)))
        hit = max(win, key=lambda i: d[i - 1])
        offs.append(hit - sh["t"] * fps)
    for sh in shots:
        for f in range(sh["start_frame"] + 1, sh["end_frame"]):
            if f < len(F):
                still.append(d[f - 1])
    offs, still = np.array(offs), np.array(still)
    rep.append(f"cut sync: mean |offset| {np.abs(offs).mean():.2f} frames, worst {np.abs(offs).max():.2f} "
               f"-> {'PASS' if np.abs(offs).max() <= 1 else 'FAIL'}")
    rep.append(f"stillness: {len(still)} held-frame pairs, max diff {still.max():.3f} -> "
               f"{'PASS' if still.max() < 1 else 'FAIL'}")
    rep.append("plan: tree/no-tree alternates (incl. loop), no transition type twice in a row, no image repeated")
    rep.append("")
    for i, sh in enumerate(shots):
        tag = "TREE" if cfg["images"][sh["image"]].get("tree") else "    "
        rep.append(f"{i + 1:2d}. {sh['t']:6.2f}s  {sh['type']:<12} {tag}  {sh['image']}"
                   + (f"  (change strength {ev_strength[sh['t']]:.2f})" if sh['t'] in ev_strength else "  (start)"))
    open(os.path.join(outdir, f"{cfg['version']}_qa.txt"), "w").write("\n".join(rep) + "\n")
    json.dump(dict(tempo=tempo, cuts=cuts, shots=shots), open(os.path.join(outdir, f"{cfg['version']}_timeline.json"), "w"), indent=1)

    # ---- sheet: every shot as it appears
    tw, th = 216, 384
    cols = 7
    rows = (len(plates) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * (tw + 12) + 12, rows * (th + 44) + 12), (40, 36, 34))
    dr = ImageDraw.Draw(sheet)
    for i, (pl, sh) in enumerate(zip(plates, shots)):
        x, y = 12 + (i % cols) * (tw + 12), 12 + (i // cols) * (th + 44)
        sheet.paste(Image.fromarray(pl).resize((tw, th)), (x, y))
        dr.text((x, y + th + 4), f"{i + 1}. {sh['t']:.2f}s {sh['type']}", fill=(235, 230, 225), font=font(13))
        dr.text((x, y + th + 22), ("TREE " if cfg["images"][sh["image"]].get("tree") else "") + sh["image"][:22],
                fill=(200, 170, 140), font=font(12))
    sheet.save(os.path.join(outdir, f"{cfg['version']}_sequence.jpg"), quality=88)
    print("\n".join(rep))


if __name__ == "__main__":
    main()
