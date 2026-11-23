#!/usr/bin/env python3
"""Centred mood-board edits: every photo, and every arrangement, sits on the screen's centre line.

    python3 centered.py --config config_ny_centered_stack.json

A config's "slides" run back to back (no blank screen between them unless a slide asks for a
fade). Each slide is one of:

  {"type": "stack", "shots": [{"image": k, "at": t, "op": "swap"|"cover", "scale": 1.0}, ...]}
      One centred photo at a time. "swap" replaces what is on screen; "cover" lands a new photo
      centred on top of the one showing, never smaller in area than it. "scale" sizes the photo
      inside the max box (1.0 = as big as the box allows).

  {"type": "inset", "base": {"image": k, "at": t, "scale": 1.0},
   "insets": [k, ...], "tree": ["v", 0, 1], "at": [t, ...], "reveal": [[0], [1]], "margin": 0.09, "gap": 22}
      A centred base photo, then smaller photos land inside it, one reveal group per time in "at"
      (default one photo each; give a mirrored pair as one group to keep the slide centred), laid out
      by "tree" (gallery.py syntax: ["h", ...] side by side, ["v", ...] stacked, numbers index
      "insets") with equal gaps and the block centred inside the base. The base shows around
      them as a frame. (Smaller photos covering a bigger one breaks the default stacked-collage
      rule; use it only when the user asks for photos landing inside a photo.)

  {"type": "sym", "images": [k, ...], "tree": [...], "reveal": [[0], [1, 2], ...],
   "at": [t, ...], "gap": 26, "margin": [60, 150]}
      A collage that is mirror-symmetric about the vertical centre line at every moment: each
      reveal group (one centred photo, or a mirrored pair of the same shape) lands at once.

An image may carry "clip": "clips/<file>.gif|mp4" (its still file in images/ is the clip's first frame, used for
layout). The clip plays from the moment it lands, at its own speed, and loops if it stays on screen longer; QA
checks that every clip stays on screen long enough to play all the way through, and checks stillness and sync with
the clips masked out. With "clips_play_once": true a clip never loops or freezes: it starts at "clip_start"
(frames to skip, e.g. a still lead-in), plays "clip_frames" frames at most (default: the rest, so a frozen tail can be
left off) and must leave the screen within them, cut at most
"clip_max_trim" frames (default 6, 0.2 s) short of its end so the next photo lands on the beat as it finishes.

Any slide may set "bg" (its background colour) and "enter": {"fade_frames": n}: the previous
slide fades out to the new background over n frames that end exactly on the slide's first
time, then the slide's first photo lands with a hard cut. Without "enter" the slide cuts in.

Times are seconds from the start of the song (frame 0 = sound 0.00 s). "end_at" is where the
music ends. Renders the MP4 (with the song), a _silent.mp4, a QA report and a sequence sheet
into out/<name>/.
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw

from gallery import spacing_check, tree_layout
from moodboard import ROOT, decode_small, font, grade, hex_rgb, load_source


# --------------------------------------------------------------------------- layout helpers
def fit(size, bw, bh):
    """Largest box at the photo's own shape inside bw x bh (never cropped, never stretched)."""
    w, h = size
    s = min(bw / w, bh / h)
    return int(round(w * s)), int(round(h * s))


def centred(w, h, cx, cy):
    return (int(round(cx - w / 2)), int(round(cy - h / 2)), w, h)


def overlaps(a, b, gap=0):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return not (ax + aw + gap <= bx or bx + bw + gap <= ax or ay + ah + gap <= by or by + bh + gap <= ay)


def touching(a, b, gap):
    """Neighbours: overlapping, or side by side / one above the other within gap + 2 px."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    side = min(ay + ah, by + bh) > max(ay, by) and max(bx - ax - aw, ax - bx - bw) <= gap + 2
    above = min(ax + aw, bx + bw) > max(ax, bx) and max(by - ay - ah, ay - by - bh) <= gap + 2
    return side or above or overlaps(a, b)


def mirror_error(boxes, W):
    """Worst mismatch between a visible photo and the mirror image of its partner across the
    vertical centre line: (px, as a share of the photo's width)."""
    worst = (0, 0.0)
    for x, y, w, h in boxes:
        mx = W - (x + w)
        e = min(abs(bx - mx) + abs(by - y) + abs(bh - h) for bx, by, bw, bh in boxes)
        if e / w > worst[1]:
            worst = (e, e / w)
    return worst


def coverage(boxes, W, H):
    m = np.zeros((H, W), bool)
    for x, y, w, h in boxes:
        m[max(y, 0):y + h, max(x, 0):x + w] = True
    return m.mean()


# --------------------------------------------------------------------------- plan
def build_states(cfg):
    """Every moment the screen changes: (time, bg, layers, enter_fade, slide index, new keys)."""
    W, H = cfg["size"]
    cx, cy = W / 2, H * cfg.get("center_y", 0.5)
    mw, mh = W * cfg["max_box"][0], H * cfg["max_box"][1]
    srcs, states, errs, info = {}, [], [], []
    kinds = {k: v.get("kind") for k, v in cfg["images"].items()}

    def src(k):
        if k not in srcs:
            if k not in cfg["images"]:
                errs.append(f"{k}: not listed in images")
            srcs[k] = load_source(cfg, k)[0]
        return srcs[k]

    for n, sl in enumerate(cfg["slides"]):
        bg = sl.get("bg", cfg["background"])
        fade = sl.get("enter", {}).get("fade_frames", 0)
        typ = sl["type"]
        first = True
        if typ == "stack":
            layers = []
            for i, sh in enumerate(sl["shots"]):
                k = sh["image"]
                w, h = fit(src(k).size, mw * sh.get("scale", 1), mh * sh.get("scale", 1))
                box = centred(w, h, cx, cy)
                op = sh.get("op", "swap")
                if i == 0 and op != "swap":
                    errs.append(f"slide {n + 1}: a stack starts with a swap")
                if op == "swap":
                    layers = [(k, box)]
                elif op == "cover":
                    pk, pb = layers[-1]
                    if w * h < pb[2] * pb[3]:
                        errs.append(f"slide {n + 1}: {k} would be smaller than the photo it covers ({pk})")
                    layers = layers + [(k, box)]
                else:
                    errs.append(f"slide {n + 1}: unknown op {op}")
                states.append(dict(t=sh["at"], bg=bg, layers=list(layers), fade=fade if first else 0, slide=n, new=[k], op=op))
                first = False
            info.append(dict(type="stack", boxes=[b for _, b in layers]))
        elif typ == "inset":
            b = sl["base"]
            bk = b["image"]
            w, h = fit(src(bk).size, mw * b.get("scale", 1), mh * b.get("scale", 1))
            base = centred(w, h, cx, cy)
            states.append(dict(t=b["at"], bg=bg, layers=[(bk, base)], fade=fade, slide=n, new=[bk], op="swap"))
            m = int(round(min(w, h) * sl.get("margin", 0.09)))
            g = sl.get("gap", 22)
            sizes = [src(k).size for k in sl["insets"]]
            # tree_layout keeps >= g to the area edge; grow the area by g so the frame is m wide
            aw, ah = w - 2 * m + 2 * g, h - 2 * m + 2 * g
            ib, _ = tree_layout(sl["tree"], sizes, aw, ah, g)
            ox, oy = base[0] + m - g, base[1] + m - g
            ib = [(x + ox, y + oy, bw, bh) for x, y, bw, bh in ib]
            for k, bb in zip(sl["insets"], ib):
                if not (bb[0] >= base[0] and bb[1] >= base[1] and bb[0] + bb[2] <= base[0] + w and bb[1] + bb[3] <= base[1] + h):
                    errs.append(f"slide {n + 1}: inset {k} spills outside the base photo")
            # reveal groups: insets that land together (a mirrored pair keeps the slide centred)
            reveal = sl.get("reveal", [[j] for j in sl.get("order", range(len(ib)))])
            layers = [(bk, base)]
            for t, grp in zip(sl["at"], reveal):
                layers = layers + [(sl["insets"][j], ib[j]) for j in grp]
                states.append(dict(t=t, bg=bg, layers=list(layers), fade=0, slide=n, new=[sl["insets"][j] for j in grp], op="inset"))
            bad = spacing_check(ib, g)
            if bad:
                errs.append(f"slide {n + 1}: uneven inset spacing {bad}")
            info.append(dict(type="inset", boxes=[base], insets=ib, margin=m, gap=g))
        elif typ == "sym":
            g = sl.get("gap", 26)
            mx, my = sl.get("margin", [60, 150])
            sizes = [src(k).size for k in sl["images"]]
            aw, ah = W - 2 * mx + 2 * g, H - 2 * my + 2 * g
            bx, _ = tree_layout(sl["tree"], sizes, aw, ah, g)
            ox, oy = (W - aw) / 2, cy - ah / 2
            boxes = [(int(round(x + ox)), int(round(y + oy)), bw, bh) for x, y, bw, bh in bx]
            layers = []
            for t, grp in zip(sl["at"], sl["reveal"]):
                layers = layers + [(sl["images"][j], boxes[j]) for j in grp]
                states.append(dict(t=t, bg=bg, layers=list(layers), fade=fade if first else 0, slide=n,
                                   new=[sl["images"][j] for j in grp], op="sym"))
                first = False
            if sorted(j for grp in sl["reveal"] for j in grp) != list(range(len(boxes))):
                errs.append(f"slide {n + 1}: reveal groups must use every photo once")
            bad = spacing_check(boxes, g)
            if bad:
                errs.append(f"slide {n + 1}: uneven spacing {bad}")
            for i in range(len(boxes)):
                for j in range(i + 1, len(boxes)):
                    if overlaps(boxes[i], boxes[j], g - 1):
                        errs.append(f"slide {n + 1}: {sl['images'][i]} and {sl['images'][j]} closer than {g}px")
            info.append(dict(type="sym", boxes=boxes, gap=g))
        else:
            errs.append(f"slide {n + 1}: unknown type {typ}")

    # ---- checks that span the whole plan
    used = [k for s in states for k in s["new"]]
    dup = sorted({k for k in used if used.count(k) > 1})
    if dup and not cfg.get("allow_repeats"):
        errs.append(f"photos used twice: {dup}")
    for i, s in enumerate(states):
        for _, (x, y, w, h) in s["layers"]:
            if x < 0 or y < 0 or x + w > W or y + h > H:
                errs.append(f"step {i + 1}: a photo runs off the screen")
        if i and s["t"] <= states[i - 1]["t"]:
            errs.append(f"step {i + 1}: time {s['t']} is not after the previous step")
    # same kind never touching: a new photo vs everything it touches, and vs the photo it replaces
    kind_errs = []
    prev_vis = []
    for i, s in enumerate(states):
        vis = s["layers"]
        g = 26
        for k in s["new"]:
            kb = dict(vis)[k]
            for ok, obox in vis:
                if ok != k and touching(kb, obox, g) and kinds.get(k) and kinds.get(k) == kinds.get(ok):
                    kind_errs.append(f"step {i + 1}: {k} touches {ok} (both {kinds[k]})")
        last = states[i - 1]["new"] if i else []
        # a fresh single photo replacing a single photo: the two are back to back
        if prev_vis and s["layers"][0][0] not in dict(prev_vis) and len(s["new"]) == 1 and len(last) == 1:
            for k in s["new"]:
                for ok in last:
                    if kinds.get(k) and kinds.get(k) == kinds.get(ok):
                        kind_errs.append(f"step {i + 1}: {k} follows {ok} (both {kinds[k]})")
        prev_vis = vis
    loop_a, loop_b = states[-1]["new"], states[0]["new"]
    for a in loop_a:
        for b in loop_b:
            if kinds.get(a) and kinds.get(a) == kinds.get(b):
                kind_errs.append(f"loop: last photo {a} and first photo {b} are both {kinds[a]}")
    errs += kind_errs
    return states, srcs, info, errs


# --------------------------------------------------------------------------- render
def load_clip(path, fps):
    """Every frame of a GIF or video, resampled to the edit's frame rate (uint8, native size)."""
    w, h = (int(v) for v in subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                                            "stream=width,height", "-of", "csv=p=0", path],
                                           capture_output=True, text=True, check=True).stdout.split(",")[:2])
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-vf", f"fps={fps}", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3)


def border(arr, px, rgb):
    if px <= 0:
        return arr
    a = arr.copy()
    a[:px], a[-px:], a[:, :px], a[:, -px:] = rgb, rgb, rgb, rgb
    return a


def plate(cfg, state, srcs, cache):
    W, H = cfg["size"]
    canvas = np.empty((H, W, 3), np.float32)
    canvas[:] = hex_rgb(state["bg"])
    bpx = cfg.get("border", {}).get("px", 0)
    brgb = hex_rgb(cfg.get("border", {}).get("color", "#FFFFFF"))
    for k, (x, y, w, h) in state["layers"]:
        if (k, w, h) not in cache:
            im = srcs[k].resize((w, h), Image.LANCZOS)  # whole photo, own shape
            cache[(k, w, h)] = border(grade(np.asarray(im), cfg["grade"]), bpx, brgb)
        canvas[y:y + h, x:x + w] = cache[(k, w, h)]
    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--plan", action="store_true", help="check the plan and print it, no render")
    cfg = json.load(open(ap.parse_args().config))
    args = ap.parse_args()
    W, H = cfg["size"]
    fps = cfg["fps"]
    states, srcs, info, errs = build_states(cfg)
    if errs:
        sys.exit("plan errors:\n  " + "\n  ".join(errs))
    if states[0]["t"] != 0:
        sys.exit("the first photo must be on screen at 0.00 s")
    N = int(round(cfg["end_at"] * fps))
    for s in states:
        s["f0"] = int(round(s["t"] * fps))
    for a, b in zip(states, states[1:] + [dict(f0=N)]):
        a["f1"] = b["f0"]
    if args.plan:
        for i, s in enumerate(states):
            print(f"{i + 1:2d} {s['t']:5.2f}s f{s['f0']:3d} slide {s['slide'] + 1} {s['op']:<5} {', '.join(s['new'])}")
        return

    outdir = os.path.join(ROOT, "out", cfg["name"])
    os.makedirs(outdir, exist_ok=True)
    cache = {}
    plates = [plate(cfg, s, srcs, cache) for s in states]
    grain = np.random.default_rng(cfg["grade"].get("grain_seed", 0)).normal(0, cfg["grade"]["grain"], (H, W, 1)).astype(np.float32)
    mp4 = os.path.join(outdir, f"{cfg['version']}.mp4")
    silent = os.path.join(outdir, f"{cfg['version']}_silent.mp4")
    song = os.path.join(ROOT, cfg["song"])
    dur = N / fps
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
           "-r", str(fps), "-i", "-", "-i", song, "-filter_complex",
           f"[1:a]atrim=0:{dur:.4f},asetpts=N/SR/TB,afade=t=out:st={dur - 0.05:.4f}:d=0.05[a]",
           "-map", "0:v", "-map", "[a]", "-c:v", "libx264", "-profile:v", "high", "-pix_fmt", "yuv420p",
           "-crf", "18", "-preset", "veryfast", "-r", str(fps), "-c:a", "aac", "-b:a", "192k",
           "-movflags", "+faststart", "-t", f"{dur:.4f}", mp4]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    # clips: frames at the edit's rate, and the frame each clip landed on (it plays from there)
    clips = {k: load_clip(os.path.join(ROOT, v["clip"]), fps) for k, v in cfg["images"].items() if v.get("clip") and k in srcs}
    landed = []
    for i, st in enumerate(states):
        lk = {}
        for k, _ in st["layers"]:
            lk[k] = landed[i - 1][k] if i and k in landed[i - 1] and k in dict(states[i - 1]["layers"]) else st["f0"]
        landed.append(lk)
    clip_cache = {}
    bpx = cfg.get("border", {}).get("px", 0)
    brgb = hex_rgb(cfg.get("border", {}).get("color", "#FFFFFF"))

    def clip_frame(k, idx, w, h):
        if (k, idx, w, h) not in clip_cache:
            im = Image.fromarray(clips[k][idx]).resize((w, h), Image.LANCZOS)
            clip_cache[(k, idx, w, h)] = border(grade(np.asarray(im), cfg["grade"]), bpx, brgb)
        return clip_cache[(k, idx, w, h)]

    once = cfg.get("clips_play_once", False)
    clip_boxes = [[] for _ in range(N)]   # per output frame: boxes of clips that are moving
    clip_seen = {k: 0 for k in clips}     # longest unbroken run on screen, in frames
    fade_frames = set()
    frames8 = []
    for i, (s, pl) in enumerate(zip(states, plates)):
        held = np.clip(pl + grain, 0, 255).astype(np.uint8)
        frames8.append(held)
        nxt = states[i + 1] if i + 1 < len(states) else None
        nf = nxt["fade"] if nxt else 0
        fade_start = s["f1"] - nf
        hb = held.tobytes()
        live = [(k, b, landed[i][k]) for k, b in s["layers"] if k in clips]
        for k, _, l0 in live:
            clip_seen[k] = max(clip_seen[k], s["f1"] - l0)
        for f in range(s["f0"], s["f1"]):
            base = pl
            if live:
                base = pl.copy()
                for k, (x, y, w, h), l0 in live:
                    if once:
                        idx = min(cfg["images"][k].get("clip_start", 0) + f - l0, len(clips[k]) - 1)
                    else:
                        idx = (f - l0) % len(clips[k])
                    base[y:y + h, x:x + w] = clip_frame(k, idx, w, h)
                    if f > l0:  # the frame it lands on is a real cut; after that its motion is left out of QA
                        clip_boxes[f].append((x, y, w, h))
            if nf and f >= fade_start:  # ease out to the next slide's background, finishing on its beat
                a = (f - fade_start + 1) / (nf + 1)
                a = a * a * (3 - 2 * a)
                target = np.empty_like(pl)
                target[:] = hex_rgb(nxt["bg"])
                fr = np.clip(base * (1 - a) + target * a + grain, 0, 255).astype(np.uint8)
                p.stdin.write(fr.tobytes())
                fade_frames.add(f)
            elif live:
                p.stdin.write(np.clip(base + grain, 0, 255).astype(np.uint8).tobytes())
            else:
                p.stdin.write(hb)
    p.stdin.close()
    if p.wait():
        sys.exit("ffmpeg failed")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", mp4, "-an", "-c:v", "copy", "-movflags",
                    "+faststart", silent], check=True)

    # ---- QA on the rendered file
    F = decode_small(mp4)
    sc = F.shape[2] / W
    d = np.empty(len(F) - 1)
    for f in range(1, len(F)):
        diff = np.abs(F[f].astype(np.float32) - F[f - 1])
        boxes = (clip_boxes[f] if f < N else []) + (clip_boxes[f - 1] if f - 1 < N else [])
        if boxes:  # moving clips are not a cut and not a wobble: leave them out of the measure
            m = np.ones(diff.shape[:2], bool)
            for x, y, w, h in boxes:
                m[max(int(y * sc) - 2, 0):int((y + h) * sc) + 2, max(int(x * sc) - 2, 0):int((x + w) * sc) + 2] = False
            d[f - 1] = diff[m].mean() if m.any() else 0.0
        else:
            d[f - 1] = diff.mean()
    offs = []
    for s in states[1:]:
        f = s["f0"]
        hit = max(range(max(f - 3, 1), min(f + 4, len(F))), key=lambda i: d[i - 1])
        offs.append(hit - s["t"] * fps)
    offs = np.abs(np.array(offs))
    still = [d[f - 1] for s in states for f in range(s["f0"] + 1, s["f1"])
             if f < len(F) and f not in fade_frames and f - 1 not in fade_frames]
    shapes_ok = all(abs(w / h - srcs[k].width / srcs[k].height) < 0.01 for s in states for k, (_, _, w, h) in s["layers"])
    kinds = {k: v.get("kind") for k, v in cfg["images"].items()}
    rep = [f"{len(states)} steps, {len(srcs)} photos, {len(cfg['slides'])} slides, {dur:.2f} s (music ends {cfg['end_at']:.2f} s)",
           f"sync: every change within {offs.max():.2f} frames of its beat (mean {offs.mean():.2f}) -> {'PASS' if offs.max() <= 1 else 'FAIL'}",
           f"stillness: max held-frame change {max(still):.3f} -> {'PASS' if max(still) < 1 else 'FAIL'}",
           f"shape: every photo whole, at its own width-to-height shape -> {'PASS' if shapes_ok else 'FAIL'}",
           "repeats: none -> PASS",
           "variety: no two photos of the same kind touching, back to back, or across the loop -> PASS",
           "position: every photo and every arrangement centred on the screen's vertical centre line"]
    if clips and once:
        lines, bad = [], []
        for k in clips:
            meta = cfg["images"][k]
            avail = meta.get("clip_frames", len(clips[k]) - meta.get("clip_start", 0))
            seen, cut = clip_seen[k], avail - clip_seen[k]
            lines.append(f"  {k}: {avail / fps:.2f} s of clip, on screen {seen / fps:.2f} s, {max(cut, 0)} frame(s) trimmed off the end")
            if cut < 0 or cut > cfg.get("clip_max_trim", 6):
                bad.append(k)
        rep.append("clips: each plays once from its first moving frame, never loops, never freezes, and the next photo lands as it "
                   f"ends -> {'PASS' if not bad else 'FAIL ' + str(bad)}")
        rep += lines
    elif clips:
        short = [k for k in clips if clip_seen[k] < len(clips[k])]
        rep.append("clips: " + ", ".join(f"{k} {len(clips[k]) / fps:.2f} s, on screen {clip_seen[k] / fps:.2f} s" for k in clips)
                   + (" -> PASS (every clip plays all the way through)" if not short else f" -> FAIL (cut short: {short})"))
    sym_worst = (0, 0.0)
    for s in states:
        if s["op"] == "sym":
            sym_worst = max(sym_worst, mirror_error([b for _, b in s["layers"]], W), key=lambda e: e[1])
    if any(i["type"] == "sym" for i in info):
        rep.append(f"symmetry: mirror-symmetric at every moment; worst mismatch {sym_worst[0]} px "
                   f"({sym_worst[1]:.1%} of that photo's width) -> {'PASS' if sym_worst[1] <= 0.03 else 'FAIL'}")
    rep.append("")
    rep.append("screen covered by photos, per slide (finished slide):")
    for n, inf in enumerate(info):
        last = [s for s in states if s["slide"] == n][-1]
        cov = coverage([b for _, b in last["layers"]], W, H)
        extra = ""
        if inf["type"] == "inset":
            extra = f"  (frame of base photo {inf['margin']} px, gap {inf['gap']} px)"
        elif inf["type"] == "sym":
            extra = f"  (gap {inf['gap']} px)"
        rep.append(f"  slide {n + 1:2d} {inf['type']:<5} {cov:5.0%}{extra}")
    rep.append("")
    for i, s in enumerate(states):
        rep.append(f"{i + 1:2d}. {s['t']:5.2f}s  slide {s['slide'] + 1:<2} {s['op']:<5} " +
                   ", ".join(f"{k} [{kinds.get(k)}]" for k in s["new"]) + (f"  (bg {s['bg']}, fade in {s['fade']}f)" if s["fade"] else ""))
    open(os.path.join(outdir, f"{cfg['version']}_qa.txt"), "w").write("\n".join(rep) + "\n")

    tw, th, cols = 180, 320, 8
    rows = (len(frames8) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * (tw + 10) + 10, rows * (th + 28) + 10), (30, 26, 26))
    dr = ImageDraw.Draw(sheet)
    for i, (fr, s) in enumerate(zip(frames8, states)):
        x, y = 10 + (i % cols) * (tw + 10), 10 + (i // cols) * (th + 28)
        sheet.paste(Image.fromarray(fr).resize((tw, th), Image.LANCZOS), (x, y))
        dr.text((x, y + th + 4), f"{i + 1}. {s['t']:.2f}s {s['op']}", fill=(235, 225, 225), font=font(12))
    sheet.save(os.path.join(outdir, f"{cfg['version']}_sequence.jpg"), quality=88)
    print("\n".join(rep))


if __name__ == "__main__":
    main()
