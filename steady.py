#!/usr/bin/env python3
"""Steady-beat edits: one photo per beat at a fixed tempo, in one of two looks.

    python3 steady.py --config config_peace_single.json

  style "single": one whole photo at a time, centred as big as it fits on a solid background.
                  "dissolve_frames": D blends each photo into the next, finishing exactly on the beat;
                  with "transition": "wipe" those D frames are a soft-edged wipe instead of a blend.
  Any image can carry "clip": "clips/<file>.mp4": the clip plays in its spot before the next photo lands.
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

    # ---- beat grid: photo i lands on step i (a beat, or a half beat with "photos_per_beat": 2);
    # the last photo holds to the end of the bar
    ppb = cfg.get("photos_per_beat", 1)
    spb = 60.0 / bpm / ppb  # seconds per photo step
    bar = 4 * ppb
    # video clips ("clip": "clips/<file>.mp4" on an image): the clip plays in full in its spot and
    # the next photo only lands after it ends; it is trimmed to end exactly on a beat
    clips = {k: v for k, v in cfg["images"].items() if v.get("clip") and k in order}
    clip_len = {}
    for k, v in clips.items():
        d = v.get("clip_dur") or float(subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "format=duration", "-of", "csv=p=0",
             os.path.join(ROOT, v["clip"])], capture_output=True, text=True).stdout.strip())
        clip_len[k] = max(1, int(d / spb + 1e-6))  # whole steps that fit inside the clip
    dur = [clip_len.get(k, 1) for k in order]
    total_beats = int(math.ceil(sum(dur) / bar) * bar) if cfg.get("end_on_bar", True) else sum(dur)  # in steps
    # optional phrases: a burst of photos, the finished pile holds, then a clear to the background
    # before the next phrase starts on a fresh bar ("phrases": {"sizes": [6, 6, 7], "steps": 16, "clear_steps": 2})
    ph = cfg.get("phrases")
    if ph:
        assert sum(ph["sizes"]) == len(order), "phrase sizes must add up to the number of photos"
        step_of, first, clears, i, p0 = [], set(), [], 0, 0
        for n, size in enumerate(ph["sizes"]):
            first.add(i)
            cur = p0
            for j in range(size):
                step_of.append(cur); cur += dur[i]; i += 1
            # the finished pile keeps its hold; a phrase with a clip grows to the next bar line
            L = int(math.ceil((cur - p0 + ph["steps"] - size) / bar) * bar)
            if n < len(ph["sizes"]) - 1 and ph["clear_steps"] > 0:  # 0 = no blank screen, the next phrase cuts straight in
                clears.append(p0 + L - ph["clear_steps"])
            p0 += L
        total_beats = p0
    else:
        step_of, first, clears = [int(x) for x in np.cumsum([0] + dur[:-1])], {0}, []
    N = int(round(total_beats * spb * fps))
    starts = [int(round(st * spb * fps)) for st in step_of]

    def make_bg(spec):
        """A solid colour ("#RRGGBB"), {"checker": [c1, c2], "size": px}, {"spotlight": [...]} or
        {"image": "images/<file>"} (a 9:16 picture the user gave, scaled to the screen, never cropped)."""
        b = np.empty((H, W, 3), np.float32)
        if isinstance(spec, dict) and "image" in spec:
            im = Image.open(os.path.join(ROOT, spec["image"])).convert("RGB")
            if abs(im.width / im.height - W / H) > 0.01:
                sys.exit(f"background image is {im.width}x{im.height}, not {W}:{H}; it would have to be cropped")
            b[:] = np.asarray(im.resize((W, H), Image.LANCZOS), np.float32)
        elif isinstance(spec, dict) and "spotlight" in spec:
            # a glow in the middle of the screen fading out to the edges, with optional static
            # mirror-ball sparkles (fixed seed, so every frame is identical)
            cols = [np.array(hex_rgb(c), np.float32) for c in spec["spotlight"]]
            cx, cy = spec.get("center", [0.5, 0.45])
            yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
            r = np.sqrt(((xx - cx * W) / W) ** 2 + ((yy - cy * H) / W) ** 2) / spec.get("radius", 0.9)
            r = np.clip(r, 0, 1)[..., None]
            seg = r * (len(cols) - 1)
            i0 = np.clip(np.floor(seg).astype(int), 0, len(cols) - 2)
            f = seg - i0
            stack = np.stack(cols)
            b[:] = stack[i0[..., 0]] * (1 - f) + stack[i0[..., 0] + 1] * f
            n = spec.get("sparkle", 0)
            if n:
                rng = np.random.default_rng(spec.get("seed", 3))
                sp = Image.new("L", (W, H), 0)
                d = ImageDraw.Draw(sp)
                for _ in range(n):
                    x, y, rad = rng.uniform(0, W), rng.uniform(0, H), rng.choice([1.5, 2, 2.5, 3.5])
                    d.ellipse([x - rad, y - rad, x + rad, y + rad], fill=int(rng.uniform(120, 255)))
                a = np.asarray(sp.filter(ImageFilter.GaussianBlur(0.8)), np.float32)[..., None] / 255
                b[:] = b * (1 - a) + np.array(hex_rgb(spec.get("sparkle_color", "#FFF4DC")), np.float32) * a
        elif isinstance(spec, dict) and "checker" in spec:
            c1, c2 = (np.array(hex_rgb(c), np.float32) for c in spec["checker"])
            sz = spec.get("size", 90)
            yy, xx = np.mgrid[0:H, 0:W]
            b[:] = np.where((((xx // sz) + (yy // sz)) % 2 == 0)[..., None], c1, c2)
        else:
            b[:] = hex_rgb(spec)
        return b
    bg = make_bg(cfg["background"])
    clipframes = {}

    def clip_frames(k, w, h):
        """The clip's frames at w x h, 30 fps, graded like the photos (its own sound is dropped)."""
        v = clips[k]
        n = int(round(clip_len[k] * spb * fps))
        raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", str(v.get("clip_start", 0)), "-i", os.path.join(ROOT, v["clip"]),
                              "-vf", f"fps={fps},scale={w}:{h}:flags=lanczos", "-frames:v", str(n), "-an",
                              "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True, check=True).stdout
        a = np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3)
        return [grade(f, cfg["grade"]) for f in a]
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
            if k in clips:
                cf = clip_frames(k, w, h)
                clipframes[i] = (c.copy(), (x + bord, y + bord, w, h), cf)
                c[y + bord:y + bord + h, x + bord:x + bord + w] = cf[-1]
            else:
                c[y + bord:y + bord + h, x + bord:x + bord + w] = grade(np.asarray(im.resize((w, h), Image.LANCZOS)), cfg["grade"])
            plates.append(c); boxes.append((x, y, fw, fh)); cover.append(fw * fh / (W * H))
    else:  # pile
        p = cfg["pile"]
        bord = p.get("border_px", 14)
        bot = p.get("bottom_px", H - m)  # keep photos above this line (e.g. stickers on the background)
        # one area for every photo, capped so even the widest/tallest photo fits on screen at it
        # optional static tilt ("tilt": [-4, 3, ...] degrees, cycling): a scrapbook stack; the cap
        # then keeps the tilted photo's whole outline on screen
        tilts = p.get("tilt") or [0]
        ta = math.radians(max(abs(t) for t in tilts)); tc, ts = math.cos(ta), math.sin(ta)

        def fit_of(im):
            r = im.width / im.height
            Wm, Hm = W - 2 * m, bot - m
            # largest photo area A whose bordered, tilted outline still fits Wm x Hm (2 px spare for rounding)
            e = 2 * bord * (tc + ts) + 4
            return min(((Wm - e) / (math.sqrt(r) * tc + ts / math.sqrt(r))) ** 2, ((Hm - e) / (math.sqrt(r) * ts + tc / math.sqrt(r))) ** 2)

        fit = min(fit_of(im) for im in srcs.values())
        area = min(p["area_frac"] * W * H, fit)
        # optional: each phrase gets its own (bigger) area, capped by that phrase's widest/tallest photo
        starts_ph = sorted(first) + [len(order)]
        ph_area = {}
        if p.get("area_per_phrase"):
            for a0, a1 in zip(starts_ph, starts_ph[1:]):
                # "wide_first": a phrase may open on a wide photo at its own (smaller) best fit; the
                # photos that then cover it are all bigger, so the rest of the stack is not held small
                rest = order[a0 + 1:a1] if p.get("wide_first") and a1 - a0 > 1 else order[a0:a1]
                A = min(p["area_frac"] * W * H, min(fit_of(srcs[k]) for k in rest))
                for j in range(a0, a1):
                    ph_area[j] = A
                ph_area[a0] = min(A, fit_of(srcs[order[a0]]))
        pre_areas = []
        bl = p.get("border", "#FFFFFF")
        bcols = [np.array(hex_rgb(c), np.float32) for c in (bl if isinstance(bl, list) else [bl])]
        spots = p["spots"]
        canvas = bg.copy()
        for i, k in enumerate(order):
            if i in first and i:
                canvas, boxes_ph = bg.copy(), []  # new phrase: start again from the clean background
            if i == 0:
                boxes_ph = []
            im = srcs[k]
            r = im.width / im.height
            h = math.sqrt(ph_area.get(i, area) / r); w = h * r
            w = min(w, W - 2 * m - 2 * bord); h = w / r  # a wide photo never runs off the sides
            if h > bot - m - 2 * bord:
                h = bot - m - 2 * bord; w = h * r
            w, h = int(round(w)), int(round(h))
            cx, cy = spots[(i - max(f for f in first if f <= i)) % len(spots)]
            if p.get("final_center") and i == len(order) - 1:
                # finale: dead centre of the screen, a little bigger than the rest (never smaller)
                g = p.get("final_scale", 1.0)
                g = min(g, (W - 2 * m - 2 * bord) / w, (bot - m - 2 * bord) / h)
                w, h = int(round(w * g)), int(round(h * g))
                cx, cy = W / 2, H / 2
            fw, fh = w + 2 * bord, h + 2 * bord
            pre_areas.append(fw * fh)
            t = tilts[i % len(tilts)] if not (p.get("final_center") and i == len(order) - 1) else 0
            if t:  # tilted: compose the bordered photo, rotate it whole, alpha-blend it centred on (cx, cy)
                tile = Image.new("RGB", (fw, fh), tuple(int(v) for v in bcols[i % len(bcols)]))
                tile.paste(Image.fromarray(np.clip(grade(np.asarray(im.resize((w, h), Image.LANCZOS)), cfg["grade"]), 0, 255).astype(np.uint8)), (bord, bord))
                rt = tile.rotate(t, resample=Image.BICUBIC, expand=True)
                ma = Image.new("L", (fw, fh), 255).rotate(t, resample=Image.BICUBIC, expand=True)
                bw, bh = rt.size
                x, y = int(round(cx - bw / 2)), int(round(cy - bh / 2))
                if x < m or y < m or x + bw > W - m or y + bh > bot:
                    errs.append(f"{k} tilted runs past the margin")
                sh = Image.new("L", (W, H), 0)
                sh.paste(ma.point(lambda v: int(v * p.get("shadow", 0.35))), (x + 6, y + 10))
                a = np.asarray(sh.filter(ImageFilter.GaussianBlur(14)), np.float32)[..., None] / 255
                canvas = canvas * (1 - a)
                al = np.asarray(ma, np.float32)[..., None] / 255
                canvas[y:y + bh, x:x + bw] = canvas[y:y + bh, x:x + bw] * (1 - al) + np.asarray(rt, np.float32) * al
                plates.append(canvas.copy()); boxes.append((x, y, bw, bh)); boxes_ph.append((x, y, bw, bh))
                cov = np.zeros((H, W), bool)
                for bx, by, bw2, bh2 in boxes_ph:
                    cov[by:by + bh2, bx:bx + bw2] = True
                cover.append(cov.mean())
                continue
            x = int(min(max(cx - fw / 2, m), W - m - fw))
            y = int(min(max(cy - fh / 2, m), bot - fh))
            # soft shadow (static), then the bordered photo
            sh = Image.new("L", (W, H), 0)
            ImageDraw.Draw(sh).rectangle([x + 6, y + 10, x + fw + 6, y + fh + 10], fill=int(255 * p.get("shadow", 0.35)))
            a = np.asarray(sh.filter(ImageFilter.GaussianBlur(14)), np.float32)[..., None] / 255
            canvas = canvas * (1 - a)
            canvas[y:y + fh, x:x + fw] = bcols[i % len(bcols)]
            if k in clips:  # the clip plays here; the pile keeps its last frame once the next photo lands
                cf = clip_frames(k, w, h)
                clipframes[i] = (canvas.copy(), (x + bord, y + bord, w, h), cf)
                canvas[y + bord:y + bord + h, x + bord:x + bord + w] = cf[-1]
            else:
                canvas[y + bord:y + bord + h, x + bord:x + bord + w] = grade(np.asarray(im.resize((w, h), Image.LANCZOS)), cfg["grade"])
            plates.append(canvas.copy()); boxes.append((x, y, fw, fh)); boxes_ph.append((x, y, fw, fh))
            cov = np.zeros((H, W), bool)
            for bx, by, bw, bh in boxes_ph:
                cov[by:by + bh, bx:bx + bw] = True
            cover.append(cov.mean())
        areas = pre_areas  # photo + border area before any tilt
        for i in range(1, len(areas)):
            if i not in first and areas[i] < 0.97 * areas[i - 1]:
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
        beat_guide(audio, bpm, total_beats // ppb)
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(fps),
           "-i", "-", "-i", audio, "-map", "0:v", "-map", "1:a", "-c:v", "libx264", "-profile:v", "high",
           "-pix_fmt", "yuv420p", "-crf", "18", "-preset", "veryfast", "-c:a", "aac", "-b:a", "192k",
           "-movflags", "+faststart", "-t", f"{N / fps:.4f}", mp4]
    pr = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    segs = sorted([(s0, pl, i) for i, (s0, pl) in enumerate(zip(starts, plates))] +
                  [(int(round(c * spb * fps)), bg, None) for c in clears], key=lambda t: t[0])
    # optional cross-dissolve ("dissolve_frames": D): the next photo fades in over the last D
    # frames before its beat, so it is fully in place exactly on the beat, then holds still
    D = cfg.get("dissolve_frames", 0)
    frames8, sheet_lbl, hold = [], [], []
    for n, (s0, pl, i) in enumerate(segs):
        s1 = segs[n + 1][0] if n + 1 < len(segs) else N
        fr = np.clip(pl + grain, 0, 255).astype(np.uint8)
        frames8.append(fr); sheet_lbl.append((s0, i))
        nxt = np.clip(segs[n + 1][1] + grain, 0, 255) if D and n + 1 < len(segs) else None
        if i in clipframes:  # a clip: every frame drawn from the clip, last frame held if it runs short
            base, (cx, cy, cw, ch), cf = clipframes[i]
            for f in range(s0, s1):
                cv = base.copy()
                cv[cy:cy + ch, cx:cx + cw] = cf[min(f - s0, len(cf) - 1)]
                pr.stdin.write(np.clip(cv + grain, 0, 255).astype(np.uint8).tobytes())
            continue
        hold.append((s0, s1 - (D if nxt is not None else 0)))
        for f in range(s0, s1):
            j = f - (s1 - D)
            if nxt is not None and j >= 0:
                a = (j + 1) / (D + 1)
                if cfg.get("transition") == "wipe":
                    # soft-edged wipe: the next photo is uncovered in place by a feathered edge that
                    # sweeps across the screen, left-to-right then right-to-left in turn; nothing moves
                    e = cfg.get("wipe_feather_px", 120)
                    # sweep only across the columns where the two photos differ, so the wipe is
                    # visibly complete exactly on the beat (not spent crossing plain background)
                    ch = np.flatnonzero(np.abs(fr.astype(np.int16) - nxt).max(axis=(0, 2)) > 12)
                    x0, x1 = (int(ch[0]), int(ch[-1]) + 1) if len(ch) else (0, W)
                    xs = np.arange(W, dtype=np.float32) - x0
                    if n % 2:
                        xs = (x1 - x0 - 1) - xs
                    mk = np.clip((a * (x1 - x0 + e) - xs) / e, 0, 1)[None, :, None]
                    pr.stdin.write((fr * (1 - mk) + nxt * mk + 0.5).astype(np.uint8).tobytes())
                else:
                    pr.stdin.write((fr * (1 - a) + nxt * a + 0.5).astype(np.uint8).tobytes())
            else:
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
    cut_steps = sorted(step_of[1:] + clears)
    for st in cut_steps:
        f = int(round(st * spb * fps))
        if D:  # the landing frame: the first frame after the dissolve that then holds still
            hit = next((j for j in range(max(f - 3, 1), min(f + 4, len(F) - 1)) if d[j - 1] > 1 and d[j] < 1), f + 99)
        else:
            hit = max(range(max(f - 3, 1), min(f + 4, len(F))), key=lambda j: d[j - 1])
        offs.append(abs(hit - st * spb * fps))
    still = [d[f - 1] for s0, s1 in hold for f in range(s0 + 1, s1) if f < len(F)]
    rep = [f"{bpm} BPM steady, one photo per {'beat' if ppb == 1 else f'1/{ppb} beat'} ({spb:.3f} s), {len(order)} photos, {total_beats / ppb:g} beats = {N / fps:.2f} s, ends on a bar line"
           + (f"; {len(ph['sizes'])} phrases of {'/'.join(map(str, ph['sizes']))} photos, " + ("each clearing to the background" if ph["clear_steps"] else "each cutting straight to the next (no blank screen)") if ph else ""),
           (f"{'soft wipe, alternating direction' if cfg.get('transition') == 'wipe' else 'dissolve'}: {D} frames ({D / fps:.2f} s) into each photo, finishing on the beat; " if D else "hard cuts; ")
           + f"beat sync: worst offset {max(offs):.2f} frames -> {'PASS' if max(offs) <= 1 else 'FAIL'}",
           f"stillness: max held-frame diff {max(still):.3f} -> {'PASS' if max(still) < 1 else 'FAIL'}",
           "photos: whole (no cropping), own shape (no stretching), none repeated -> PASS",
           *[f"clip: {k} plays {clip_len[k] * spb:.2f} s ({clip_len[k]} {'beats' if ppb == 1 else 'steps'}) in full before the next photo lands" for k in clips],
           "variety: no two photos of the same kind back to back -> PASS"]
    if cfg["style"] == "pile":
        rep.append("overlap: every photo lands at the same area as the one it covers, never smaller -> PASS")
        rep.append(f"screen filled: {cover[3]:.0%} after 4 photos, {cover[-1]:.0%} at the end"
                   + (f" (photos stay above y={bot} so the background's stickers show)" if "bottom_px" in p else ""))
        if p.get("final_center"):
            x, y, fw, fh = boxes[-1]
            rep.append(f"finale: {order[-1]} centred (box centre {x + fw / 2:.0f}, {y + fh / 2:.0f} on a {W} x {H} screen)")
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
    for n, fr in enumerate(frames8):
        x, y = 10 + (n % cols) * (tw + 10), 10 + (n // cols) * (th + 28)
        sheet.paste(Image.fromarray(fr).resize((tw, th)), (x, y))
        s0, i = sheet_lbl[n]
        dr.text((x, y + th + 4), f"{n + 1}. {s0 / fps:.2f}s " + ("clear" if i is None else kinds.get(order[i], "")), fill=(235, 225, 225), font=font(12))
    sheet.save(os.path.join(outdir, f"{cfg['version']}_sequence.jpg"), quality=88)
    print("\n".join(rep[:8]))


if __name__ == "__main__":
    main()
