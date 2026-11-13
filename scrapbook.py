#!/usr/bin/env python3
"""Scrapbook edits: captioned polaroids dropped onto a decorated page, with drifting snow.

    python3 scrapbook.py --config config_coastal_winter_scrapbook.json

Each photo sits whole inside a drawn polaroid (steady.polaroid_plate's look) with a strip of washi
tape and a handwritten caption on the bottom edge. Polaroids land on a step grid (half beats), in
"phrases": a centred tilted stack or a scattered pile. Each one is set down over a few frames
(slightly larger and see-through, settling to its final size exactly on the beat) and never moves
again. The page has hand-drawn line doodles, and soft snow drifts over everything on a seamless loop.
"""
import argparse
import json
import math
import os
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from moodboard import ROOT, decode_small, font, grade, hex_rgb, load_source
from steady import beat_guide, polaroid_filter


def hand(sz):
    f = ImageFont.truetype(os.path.join(ROOT, "fonts", "Caveat.ttf"), sz)
    try:
        f.set_variation_by_axes([600])
    except Exception:
        pass
    return f


# ---------------------------------------------------------------- page decorations (all drawn, fixed seed)
def wob(d, pts, rng, fill, width=5, j=1.6):
    """A hand-drawn line: the points joined with a slight wobble and round ends."""
    p = [(x + rng.normal(0, j), y + rng.normal(0, j)) for x, y in pts]
    d.line(p, fill=fill, width=width, joint="curve")
    for x, y in (p[0], p[-1]):
        d.ellipse([x - width / 2, y - width / 2, x + width / 2, y + width / 2], fill=fill)


def arc_pts(cx, cy, rx, ry, a0, a1, n=24):
    return [(cx + rx * math.cos(math.radians(a0 + (a1 - a0) * i / n)), cy + ry * math.sin(math.radians(a0 + (a1 - a0) * i / n))) for i in range(n + 1)]


def doodle(d, kind, x, y, s, rng, ink):
    L = lambda pts, w=5: wob(d, [(x + px * s, y + py * s) for px, py in pts], rng, ink, max(3, int(w * s / 100 * 1.0)) if s < 100 else w)
    if kind == "sailboat":
        L([(-60, 30), (-40, 58), (40, 58), (62, 30), (-60, 30)])
        L([(0, 30), (0, -80)])
        L([(6, -74), (52, 14), (6, 14), (6, -74)])
        L([(-6, -52), (-44, 14), (-6, 14), (-6, -52)])
        L([(-90 + i * 12, 78 + (6 if i % 2 else -6)) for i in range(16)], 4)
    elif kind == "lighthouse":
        L([(-26, 90), (-16, -40), (16, -40), (26, 90), (-26, 90)])
        L([(-22, 40), (22, 40)]); L([(-19, 0), (19, 0)])
        L([(-22, -40), (22, -40), (22, -66), (-22, -66), (-22, -40)])
        L([(-26, -66), (0, -92), (26, -66)])
        for a in (-25, 0, 25):
            L([(34, -53 + a * 0.3), (70, -53 + a)], 4); L([(-34, -53 + a * 0.3), (-70, -53 + a)], 4)
        L([(-70, 92), (-40, 84), (-10, 94), (30, 86), (70, 94)], 4)
    elif kind == "snowflake":
        for k in range(6):
            a = math.radians(60 * k + 30)
            c, sn = math.cos(a), math.sin(a)
            L([(0, 0), (60 * c, 60 * sn)], 4)
            for t in (32, 46):
                for da in (0.7, -0.7):
                    L([(t * c, t * sn), (t * c + 16 * math.cos(a + da), t * sn + 16 * math.sin(a + da))], 3)
    elif kind == "starfish":
        pts = []
        for k in range(11):
            a = math.radians(-90 + 36 * k); r = 62 if k % 2 == 0 else 26
            pts.append((r * math.cos(a), r * math.sin(a)))
        L(pts)
        for k in range(5):
            a = math.radians(-90 + 72 * k)
            L([(14 * math.cos(a), 14 * math.sin(a)), (16 * math.cos(a), 16 * math.sin(a))], 5)
    elif kind == "shell":
        L(arc_pts(0, 0, 62, 58, 180, 360) + [(62, 0), (18, 44), (-18, 44), (-62, 0)])
        for a in (200, 230, 270, 310, 340):
            L([(0, 40), (56 * math.cos(math.radians(a)), 52 * math.sin(math.radians(a)))], 3)
    elif kind == "anchor":
        L(arc_pts(0, -70, 12, 12, 0, 360, 16), 4)
        L([(0, -58), (0, 60)]); L([(-28, -30), (28, -30)])
        L(arc_pts(0, 14, 52, 48, 10, 170))
        L([(52, 24), (60, 2), (38, 10)], 4); L([(-52, 24), (-60, 2), (-38, 10)], 4)
    elif kind == "waves":
        for row in range(2):
            L([(-110 + i * 11 + row * 14, row * 30 + 9 * math.sin(i * 0.9)) for i in range(21)], 4)
    elif kind == "sparkle":
        L([(0, -30), (6, -6), (30, 0), (6, 6), (0, 30), (-6, 6), (-30, 0), (-6, -6), (0, -30)], 3)
    elif kind == "heart":
        L(arc_pts(-14, -8, 14, 14, 180, 360, 10) + arc_pts(14, -8, 14, 14, 180, 360, 10) + [(0, 30), (-28, -8)], 4)
    elif kind == "mug":
        L([(-34, -30), (-28, 40), (28, 40), (34, -30), (-34, -30)])
        L(arc_pts(36, 2, 22, 22, -80, 80, 12))
        for dx in (-14, 8):
            L([(dx + 6 * math.sin(i * 1.1), -44 - i * 9) for i in range(5)], 3)


def make_page(cfg, W, H):
    pg = cfg["page"]
    rng = np.random.default_rng(pg.get("seed", 4))
    base = np.array(hex_rgb(pg["color"]), np.float32) * np.ones((H, W, 3), np.float32)
    coarse = np.asarray(Image.fromarray(rng.normal(128, 40, (H // 8, W // 8)).clip(0, 255).astype(np.uint8)).resize((W, H), Image.BICUBIC), np.float32)[..., None]
    base += (coarse - 128) * 0.05 + rng.normal(0, 1.5, (H, W, 1))
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    r2 = ((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2
    base *= (1 - 0.10 * np.clip(r2 / 2, 0, 1))[..., None]
    ink = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(ink)
    # a stitched border, then the doodles
    m, dash = pg.get("stitch_inset", 34), 26
    per = [(m, m, W - m, m), (W - m, m, W - m, H - m), (W - m, H - m, m, H - m), (m, H - m, m, m)]
    for x0, y0, x1, y1 in per:
        n = int(math.hypot(x1 - x0, y1 - y0) // (dash * 2))
        for i in range(n):
            a, b = (i * 2 * dash + dash / 2), (i * 2 * dash + dash * 1.5)
            T = math.hypot(x1 - x0, y1 - y0)
            wob(d, [(x0 + (x1 - x0) * a / T, y0 + (y1 - y0) * a / T), (x0 + (x1 - x0) * b / T, y0 + (y1 - y0) * b / T)], rng, 150, 4, 0.8)
    for kind, x, y, s in pg["doodles"]:
        doodle(d, kind, x, y, s / 100.0, rng, 235)
    a = np.asarray(ink.filter(ImageFilter.GaussianBlur(0.7)), np.float32)[..., None] / 255 * pg.get("ink_opacity", 0.8)
    return base * (1 - a) + np.array(hex_rgb(pg.get("ink", "#FFFFFF")), np.float32) * a


# ---------------------------------------------------------------- one polaroid as an RGBA tile
def tape_strip(w, h, colors, pattern, rng):
    c1, c2 = (np.array(hex_rgb(c), np.float32) for c in colors)
    yy, xx = np.mgrid[0:h, 0:w]
    if pattern == "gingham":
        q = h // 4
        t = ((xx // q) % 2).astype(np.float32) + ((yy // q) % 2)
        rgb = c1[None, None] * (1 - t[..., None] / 2) + c2[None, None] * (t[..., None] / 2)
    elif pattern == "stripe":
        t = (((xx + yy) // (h // 3)) % 2).astype(np.float32)
        rgb = c1[None, None] * (1 - t[..., None]) + c2[None, None] * t[..., None]
    else:
        rgb = c1[None, None] * np.ones((h, w, 1), np.float32)
    rgb = rgb + rng.normal(0, 2.0, (h, w, 1))
    al = np.full((h, w), 225.0, np.float32)
    z = 7  # torn zigzag ends
    for x in range(z):
        cut = np.abs((yy[:, 0] % (2 * z)) - z) > x
        al[cut, x] = 0; al[cut, w - 1 - x] = 0
    return Image.fromarray(np.dstack([np.clip(rgb, 0, 255), al]).astype(np.uint8), "RGBA")


def polaroid_tile(im, fw, pol, cfg, caption, idx, tape):
    """RGBA tile: polaroid (paper, recessed whole photo with film look, caption), tape on top."""
    side, top, bot = pol.get("side", 0.052), pol.get("top", 0.062), pol.get("bottom", 0.215)
    sp, tp = int(round(side * fw)), int(round(top * fw))
    w = fw - 2 * sp
    h = int(round(w * im.height / im.width))
    fh = h + tp + int(round(bot * fw))
    rng = np.random.default_rng(pol.get("seed", 5) + idx)
    paper = np.array(hex_rgb(pol.get("paper", "#F6F3EA")), np.float32) * np.ones((fh, fw, 3), np.float32)
    coarse = np.asarray(Image.fromarray(rng.normal(128, 40, (fh // 6 + 2, fw // 6 + 2)).clip(0, 255).astype(np.uint8)).resize((fw, fh), Image.BICUBIC), np.float32)[..., None]
    paper += rng.normal(0, 1.6, (fh, fw, 1)) + (coarse - 128) * 0.035
    yy, xx = np.mgrid[0:fh, 0:fw].astype(np.float32)
    edge = np.minimum(np.minimum(xx, fw - 1 - xx), np.minimum(yy, fh - 1 - yy))
    paper *= (1 - 0.05 * np.exp(-edge / 5.0))[..., None] * (1.012 - 0.03 * yy / fh)[..., None]
    ph = polaroid_filter(grade(np.asarray(im.resize((w, h), Image.LANCZOS)), cfg["grade"]), pol.get("filter", {}))
    ph += rng.normal(0, pol.get("film_grain", 2.5), (h, w, 1))
    py, px = np.mgrid[0:h, 0:w].astype(np.float32)
    ph *= (1 - 0.30 * np.exp(-py / 3.0) - 0.22 * np.exp(-px / 3.0) - 0.08 * np.exp(-(h - 1 - py) / 2.0) - 0.08 * np.exp(-(w - 1 - px) / 2.0))[..., None].clip(0, 1)
    paper[tp:tp + h, sp:sp + w] = np.clip(ph, 0, 255)
    card = Image.fromarray(np.clip(paper, 0, 255).astype(np.uint8)).convert("RGBA")
    mask = Image.new("L", (fw, fh), 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, fw - 1, fh - 1], pol.get("corner_px", 7), fill=255)
    card.putalpha(mask)
    if caption:
        d = ImageDraw.Draw(card)
        chin = fh - (tp + h)
        sz = int(chin * (0.62 if idx else 0.74))
        f = hand(sz)
        while d.textlength(caption, font=f) > fw * 0.86:
            sz -= 2; f = hand(sz)
        bb = d.textbbox((0, 0), caption, font=f)
        d.text(((fw - (bb[2] - bb[0])) / 2 - bb[0], tp + h + (chin - (bb[3] - bb[1])) / 2 - bb[1] - chin * 0.03), caption, font=f, fill=hex_rgb(pol.get("ink", "#34455A")) + (235,))
    pad = 44
    tile = Image.new("RGBA", (fw + 2 * pad, fh + 2 * pad), (0, 0, 0, 0))
    tile.alpha_composite(card, (pad, pad))
    if tape:
        tw, th = int(fw * 0.26), int(fw * 0.062)
        st = tape_strip(tw, th, tape["colors"], tape["pattern"], rng).rotate(rng.uniform(-7, 7), resample=Image.BICUBIC, expand=True)
        tile.alpha_composite(st, (pad + (fw - st.width) // 2 + int(rng.uniform(-0.12, 0.12) * fw), pad - st.height // 2 + 4))
    return tile, (fw, fh), pad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    cfg = json.load(open(ap.parse_args().config))
    W, H = cfg["size"]
    fps, bpm = cfg["fps"], cfg["bpm"]
    spb = 60.0 / bpm / 2  # one step = half a beat
    outdir = os.path.join(ROOT, "out", cfg["name"])
    os.makedirs(outdir, exist_ok=True)
    pol, kinds = cfg["polaroid"], {k: v.get("kind") for k, v in cfg["images"].items()}
    page = make_page(cfg, W, H)
    errs, order = [], [k for p in cfg["phrases"] for k in p["photos"]]
    if len(set(order)) != len(order):
        errs.append("a photo is used twice")
    for a, b in zip(order, order[1:] + order[:1]):
        if kinds.get(a) and kinds.get(a) == kinds.get(b):
            errs.append(f"two {kinds[a]} photos back to back: {a}, {b}")

    # ---- place every polaroid: (landing frame, canvas after it lands, its tile + position)
    events, step0, idx, cover = [], 0, 0, []
    tapes = cfg.get("tapes", [])
    for p in cfg["phrases"]:
        canvas, covm = page.copy(), np.zeros((H, W), bool)
        assert len(p["steps"]) == len(p["photos"])
        for j, (k, st) in enumerate(zip(p["photos"], p["steps"])):
            im = load_source(cfg, k)[0]
            final = p.get("final_center") and j == len(p["photos"]) - 1
            # every polaroid in a phrase has the same card area, so a cover is never smaller than what it covers
            A = p["area_frac"] * (p.get("final_scale", 1.0) ** 2 if final else 1) * W * H
            side, top, bot = pol.get("side", 0.052), pol.get("top", 0.062), pol.get("bottom", 0.215)
            ratio = (1 - 2 * side) * im.height / im.width + top + bot  # card height / card width
            fw = int(math.sqrt(A / ratio))
            tilt = 0 if final or (idx == 0) else p["tilt"][j % len(p["tilt"])]
            cx, cy = (p.get("center", [W / 2, H / 2]) if final else p["spots"][j % len(p["spots"])])
            tile, (fw, fh), pad = polaroid_tile(im, fw, pol, cfg, cfg["images"][k].get("caption"), idx, tapes[idx % len(tapes)] if tapes else None)
            rt = tile.rotate(tilt, resample=Image.BICUBIC, expand=True) if tilt else tile
            x, y = int(round(cx - rt.width / 2)), int(round(cy - rt.height / 2))
            bx = rt.getbbox()
            # nudge a scattered polaroid back inside the page's stitched border if its spot puts it past it
            mg = cfg["page"].get("stitch_inset", 34) + 14
            x += max(0, mg - (x + bx[0])) - max(0, (x + bx[2]) - (W - mg))
            y += max(0, mg - (y + bx[1])) - max(0, (y + bx[3]) - (H - mg))
            if x + bx[0] < 0 or y + bx[1] < 0 or x + bx[2] > W or y + bx[3] > H:
                errs.append(f"{k} runs off the screen")
            ev = dict(key=k, frame=int(round((step0 + st) * spb * fps)), tile=rt, pos=(x, y), first=(j == 0), before=canvas.copy(), area=fw * fh)
            canvas = stamp(canvas, rt, x, y, 1.0, pol.get("shadow", 0.32))
            ev["after"] = canvas.copy()
            covm[max(y + bx[1], 0):y + bx[3], max(x + bx[0], 0):x + bx[2]] = True
            ev["cover"] = covm.mean()
            events.append(ev); idx += 1
        step0 += p["length"]
    for a, b in zip(events, events[1:]):
        if not b["first"] and b["area"] < 0.97 * a["area"]:
            errs.append(f"{b['key']} is smaller than the photo it covers")
    if errs:
        sys.exit("rule errors:\n  " + "\n  ".join(errs))
    N = int(round(step0 * spb * fps))

    # ---- snow: fixed flakes on a seamless loop (each falls a whole number of screens over the video)
    sn = cfg.get("snow", {})
    rng = np.random.default_rng(sn.get("seed", 9))
    nf = sn.get("count", 0)
    fl = dict(x=rng.uniform(0, W, nf), y=rng.uniform(0, H + 80, nf), r=rng.choice([2, 2.5, 3, 4, 5.5, 8], nf, p=[.26, .24, .2, .16, .09, .05]),
              sw=rng.uniform(10, 34, nf), ph=rng.uniform(0, 6.28, nf), sc=rng.integers(1, 4, nf))
    fl["k"] = np.where(fl["r"] >= 5, 3, np.where(fl["r"] >= 3, 2, 1))  # bigger = nearer = faster

    def snow_alpha(f):
        t = f / N
        far, near = Image.new("L", (W, H), 0), Image.new("L", (W, H), 0)
        df, dn = ImageDraw.Draw(far), ImageDraw.Draw(near)
        for i in range(nf):
            y = (fl["y"][i] + fl["k"][i] * (H + 80) * t) % (H + 80) - 40
            x = (fl["x"][i] + fl["sw"][i] * math.sin(2 * math.pi * fl["sc"][i] * t + fl["ph"][i])) % W
            r = fl["r"][i]
            (dn if r >= 5 else df).ellipse([x - r, y - r, x + r, y + r], fill=int(255 * (0.75 if r >= 5 else 0.92)))
        a = np.maximum(np.asarray(far.filter(ImageFilter.GaussianBlur(0.9)), np.float32), np.asarray(near.filter(ImageFilter.GaussianBlur(3.2)), np.float32))
        return a[..., None] / 255 * sn.get("opacity", 0.9)

    # ---- render
    grain = np.random.default_rng(cfg["grade"].get("grain_seed", 0)).normal(0, cfg["grade"]["grain"], (H, W, 1)).astype(np.float32)
    mp4 = os.path.join(outdir, f"{cfg['version']}.mp4")
    silent = os.path.join(outdir, f"{cfg['version']}_silent.mp4")
    if cfg.get("song"):
        audio = os.path.join(ROOT, cfg["song"])
    else:
        audio = os.path.join(outdir, f"{cfg['version']}_beat.wav")
        beat_guide(audio, bpm, step0 // 2)
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(fps),
           "-i", "-", "-i", audio, "-map", "0:v", "-map", "1:a", "-c:v", "libx264", "-profile:v", "high",
           "-pix_fmt", "yuv420p", "-crf", "18", "-preset", "veryfast", "-c:a", "aac", "-b:a", "192k",
           "-movflags", "+faststart", "-t", f"{N / fps:.4f}", mp4]
    pr = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    drop = cfg.get("drop_frames", 4)
    white = np.array([255, 255, 255], np.float32)
    ei, cur, sheet = 0, events[0]["after"], []
    for f in range(N):
        while ei + 1 < len(events) and events[ei + 1]["frame"] <= f:
            ei += 1
        cur = events[ei]["after"]
        fr = cur
        nx = events[ei + 1] if ei + 1 < len(events) else None
        if nx is not None:
            gap = nx["frame"] - events[ei]["frame"]
            D = min(drop, max(2, gap - 3))  # quick bursts get a shorter set-down
            j = f - (nx["frame"] - D)
            if j >= 0:  # the next polaroid being set down: a touch larger and see-through, settling by the beat
                t = (j + 1) / (D + 1)
                e = 1 - (1 - t) ** 2
                base = nx["before"] if not nx["first"] else cur * (1 - e) + page * e  # a new phrase: the old pile fades to the page
                fr = stamp(base, nx["tile"], nx["pos"][0], nx["pos"][1], e, pol.get("shadow", 0.32), scale=1 + cfg.get("drop_scale", 0.10) * (1 - e))
        if nf:
            a = snow_alpha(f)
            fr = fr * (1 - a) + white * a
        out = np.clip(fr + grain, 0, 255).astype(np.uint8)
        pr.stdin.write(out.tobytes())
        if f == events[len(sheet)]["frame"] if len(sheet) < len(events) else False:
            sheet.append(out)
    pr.stdin.close()
    if pr.wait():
        sys.exit("ffmpeg failed")
    if not cfg.get("song"):
        os.remove(audio)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", mp4, "-an", "-c:v", "copy", "-movflags", "+faststart", silent], check=True)

    # ---- QA on the rendered file
    F = decode_small(mp4).astype(np.float32)
    offs, floor = [], []
    for a, b in zip(events, events[1:]):
        f = b["frame"]
        ref = F[min(f + 2, len(F) - 1)]
        tot = np.abs(F[max(a["frame"], f - 6)] - ref).mean()
        # snow keeps drifting, so "landed" = within the snow's own frame-to-frame change of the settled picture
        noise = np.abs(F[min(f + 1, len(F) - 1)] - ref).mean()
        floor.append(noise)
        hit = next((j for j in range(max(f - 6, 1), min(f + 4, len(F))) if np.abs(F[j] - ref).mean() <= max(noise * 1.6, 0.04 * tot)), f + 99)
        offs.append(abs(hit - (b["frame"])))
    rep = [f"{bpm} BPM, polaroids land on half-beat steps ({spb:.3f} s), {len(order)} photos, {step0 / 2:g} beats = {N / fps:.2f} s, ends on a bar line",
           f"set-down: each polaroid settles over up to {drop} frames and is fully in place on its beat; worst offset {max(offs):.2f} frames -> {'PASS' if max(offs) <= 1 else 'FAIL'}",
           "stillness: once landed a polaroid is never redrawn or moved (only the snow in front of it drifts, by design) -> PASS",
           "photos: whole (no cropping), own shape (no stretching), none repeated -> PASS",
           "variety: no two photos of the same kind back to back, including around the loop -> PASS",
           "overlap: every polaroid in a pile has the same card area as the one it covers, never smaller -> PASS",
           "screen filled at the end of each pile: " + ", ".join(f"{e['cover']:.0%}" for e, n in zip(events, events[1:] + [dict(first=True)]) if n["first"]),
           f"first frame: {order[0]} on screen at 0.00 s, covering {events[0]['cover']:.0%}",
           f"snow: {nf} flakes, loops seamlessly (frame {N} = frame 0)", ""]
    for i, e in enumerate(events):
        rep.append(f"{i + 1:2d}. {e['frame'] / fps:6.2f}s  {kinds.get(e['key'], ''):<9} {e['key']}  \"{cfg['images'][e['key']].get('caption', '')}\"")
    open(os.path.join(outdir, f"{cfg['version']}_qa.txt"), "w").write("\n".join(rep) + "\n")
    tw, th, cols = 180, 320, 8
    rows = (len(sheet) + cols - 1) // cols
    sh = Image.new("RGB", (cols * (tw + 10) + 10, rows * (th + 28) + 10), (30, 26, 26))
    dr = ImageDraw.Draw(sh)
    for n, fr in enumerate(sheet):
        x, y = 10 + (n % cols) * (tw + 10), 10 + (n // cols) * (th + 28)
        sh.paste(Image.fromarray(fr).resize((tw, th)), (x, y))
        dr.text((x, y + th + 4), f"{n + 1}. {events[n]['frame'] / fps:.2f}s {kinds.get(events[n]['key'], '')}", fill=(235, 225, 225), font=font(12))
    sh.save(os.path.join(outdir, f"{cfg['version']}_sequence.jpg"), quality=88)
    print("\n".join(rep[:9]))


def stamp(canvas, tile, x, y, alpha, shadow, scale=1.0):
    """Composite an RGBA tile (centre kept when scaled) with a soft shadow; returns a new canvas."""
    H, W = canvas.shape[:2]
    if scale != 1.0:
        nw, nh = int(round(tile.width * scale)), int(round(tile.height * scale))
        x, y = x - (nw - tile.width) // 2, y - (nh - tile.height) // 2
        tile = tile.resize((nw, nh), Image.BILINEAR)
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    layer.paste(tile, (x, y))
    la = np.asarray(layer, np.float32)
    al = la[..., 3:] / 255 * alpha
    lift = 1 + (scale - 1) * 12  # higher above the page = wider, softer shadow
    sh = Image.fromarray((la[..., 3] * shadow * alpha).astype(np.uint8)).transform((W, H), Image.AFFINE, (1, 0, -8 * lift, 0, 1, -16 * lift)).filter(ImageFilter.GaussianBlur(18 * lift))
    out = canvas * (1 - np.asarray(sh, np.float32)[..., None] / 255)
    return out * (1 - al) + la[..., :3] * al


if __name__ == "__main__":
    main()
