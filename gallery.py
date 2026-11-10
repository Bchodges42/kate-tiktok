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


def compositions(n, kmax, kmin=1):
    """Every way to split n photos (kept in order) into lines of kmin..kmax photos."""
    if n == 0:
        yield []
        return
    for k in range(kmin, min(kmax, n) + 1):
        for rest in compositions(n - k, kmax, kmin):
            yield [k] + rest


def justified(sizes, flow, W, H, g, kmax=3, kmin=1):
    """Lay photos out in lines with exactly `g` px between every neighbour (and at least g to
    the screen edge). flow ltr/rtl: rows of photos side by side, all photos in a row the same
    height; ttb/btt: columns stacked top to bottom, all photos in a column the same width.
    Every line spans the same length, so the block is a clean rectangle; the line split that
    gives the biggest photos wins, and the block is centred. Returns boxes in appearance order."""
    horiz = flow in ("ltr", "rtl")
    M, C = (W, H) if horiz else (H, W)  # main axis (along a line) and cross axis
    M, C = M - 2 * g, C - 2 * g
    r = [w / h if horiz else h / w for w, h in sizes]  # main length per unit of cross length
    best = None
    for comp in compositions(len(sizes), kmax, kmin):
        lines, i = [], 0
        for k in comp:
            lines.append(list(range(i, i + k))); i += k
        R = [sum(r[j] for j in ln) for ln in lines]
        m = M
        cross = sum((m - g * (len(ln) - 1)) / Rl for ln, Rl in zip(lines, R)) + g * (len(lines) - 1)
        if cross > C:  # too tall: shorten every line until it fits
            m = (C - g * (len(lines) - 1) + g * sum((len(ln) - 1) / Rl for ln, Rl in zip(lines, R))) / sum(1 / Rl for Rl in R)
            cross = C
        area = sum((m - g * (len(ln) - 1)) / Rl * (m - g * (len(ln) - 1)) / Rl * r[j]
                   for ln, Rl in zip(lines, R) for j in ln)
        if best is None or area > best[0] + 1e-6:
            best = (area, lines, R, m, cross)
    _, lines, R, m, cross = best
    m0 = g + round((M - m) / 2)
    c = g + round((C - cross) / 2)
    boxes = [None] * len(sizes)
    rev = flow in ("rtl", "btt")
    for ln, Rl in zip(lines, R):
        cs = (m - g * (len(ln) - 1)) / Rl
        p = 0.0
        for j in ln:
            ln_len = cs * r[j]
            a0, a1 = m0 + round(p), m0 + round(p + ln_len)
            if rev:  # start from the right / bottom
                a0, a1 = 2 * m0 + round(m) - a1, 2 * m0 + round(m) - a0
            b0, b1 = c, c + round(cs)
            boxes[j] = (a0, b0, a1 - a0, b1 - b0) if horiz else (b0, a0, b1 - b0, a1 - a0)
            p += ln_len + g
        c += round(cs) + g
    return boxes, best[0] / ((M + 2 * g) * (C + 2 * g))


def tree_layout(tree, sizes, W, H, g):
    """Nested layout with exactly g px between neighbours. A tree is a photo index or
    ["h", child, child, ...] (side by side, same height) / ["v", child, ...] (stacked, same width).
    Every block's width is an exact affine function of its height (w = a*h + b), so the whole
    arrangement scales to the biggest size that fits the screen with all gaps kept at g."""
    def ab(t):
        if isinstance(t, int):
            return sizes[t][0] / sizes[t][1], 0.0
        kids = [ab(c) for c in t[1:]]
        n = len(kids) - 1
        if t[0] == "h":
            return sum(a for a, _ in kids), sum(b for _, b in kids) + g * n
        a = 1 / sum(1 / a for a, _ in kids)
        return a, a * (sum(b / a for a, b in kids) - g * n)

    def place(t, x, y, h, out):
        a, b = ab(t)
        w = a * h + b
        if isinstance(t, int):
            out[t] = (x, y, w, h)
            return
        for c in t[1:]:
            ca, cb = ab(c)
            if t[0] == "h":
                cw = ca * h + cb
                place(c, x, y, h, out); x += cw + g
            else:
                ch = (w - cb) / ca
                place(c, x, y, ch, out); y += ch + g

    a, b = ab(tree)
    Mw, Mh = W - 2 * g, H - 2 * g
    h = min(Mh, (Mw - b) / a)
    w = a * h + b
    out = {}
    place(tree, g + (Mw - w) / 2, g + (Mh - h) / 2, h, out)
    # round edges (not sizes) so every gap stays exactly g px
    boxes = []
    for i in range(len(sizes)):
        x, y, bw, bh = out[i]
        x0, y0 = round(x), round(y)
        boxes.append((x0, y0, round(x + bw) - x0, round(y + bh) - y0))
    cov = sum(bw * bh for _, _, bw, bh in boxes) / (W * H)
    return boxes, cov


def spacing_check(boxes, g):
    """Every pair of neighbouring photos (sharing a row or column band) must be exactly g apart."""
    bad = []
    for i, a in enumerate(boxes):
        for b in boxes[i + 1:]:
            ax, ay, aw, ah = a
            bx, by, bw, bh = b
            if min(ay + ah, by + bh) > max(ay, by):  # side by side
                gx = max(bx - (ax + aw), ax - (bx + bw))
                if gx != g and not any(o not in (a, b) and min(o[1] + o[3], ay + ah) > max(o[1], ay)
                                       and min(ax, bx) < o[0] < max(ax, bx) for o in boxes):
                    bad.append(gx)
            if min(ax + aw, bx + bw) > max(ax, bx):  # one above the other
                gy = max(by - (ay + ah), ay - (by + bh))
                if gy != g and not any(o not in (a, b) and min(o[0] + o[2], ax + aw) > max(o[0], ax)
                                       and min(ay, by) < o[1] < max(ay, by) for o in boxes):
                    bad.append(gy)
    return bad


def timeline_from_groups(cfg, W, H):
    """Turn cfg["groups"] into timeline steps with fixed boxes (one photo per step, clear between groups)."""
    g, steps = cfg["min_gap_px"], []
    cfg["_coverage"], cfg["_group_boxes"] = [], []
    for n, grp in enumerate(cfg["groups"]):
        sizes = [load_source(cfg, k)[0].size for k in grp["images"]]
        if "tree" in grp:  # nested arrangement; photos appear in the order they're listed
            boxes, cov = tree_layout(grp["tree"], sizes, W, H, g)
        else:
            boxes, cov = justified(sizes, grp["flow"], W, H, g, grp.get("max_per_line", 3), grp.get("min_per_line", 1))
        cfg["_coverage"].append(cov)
        cfg["_group_boxes"].append(boxes)
        holds = grp.get("holds", {})
        order = grp.get("order", list(range(len(boxes))))  # appearance order (indices into images)
        for k, b in ((grp["images"][j], boxes[j]) for j in order):
            steps.append({"image": k, "slot": f"g{n + 1}-{grp.get('name', grp.get('flow'))}", "box": list(b), "beats": holds.get(k, cfg["step_beats"])})
        if n < len(cfg["groups"]) - 1:
            steps.append({"clear": True, "beats": cfg["step_beats"]})
    return steps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    cfg = json.load(open(ap.parse_args().config))
    if "groups" in cfg:
        cfg["timeline"] = timeline_from_groups(cfg, *cfg["size"])
    fps = cfg["fps"]
    W, H = cfg["size"]
    outdir = os.path.join(ROOT, "out", cfg["name"])
    os.makedirs(outdir, exist_ok=True)
    slots = cfg.get("slots", {})
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

    # ---- sizes: either fitted into each slot's w x h box, or (auto_grow) every photo in a
    # group grows together, in small steps, until it is `gap` px from a neighbour or the edge
    srcs, layers, plates, errs = {}, [], [], []
    gap = cfg.get("min_gap_px", 40)
    for sh in shots:
        if not sh.get("clear") and sh["image"] not in srcs:
            srcs[sh["image"]] = load_source(cfg, sh["image"])[0]
    grown = {}
    if cfg.get("auto_grow"):
        margin = cfg["auto_grow"].get("margin_px", gap)
        groups, cur = [], []
        for i, sh in enumerate(shots):
            if sh.get("clear"):
                groups.append(cur); cur = []
            else:
                cur.append(i)
        groups.append(cur)

        def box_of(i, hgt):
            """Box for photo i at height hgt. 'anchor' pins a side/corner to the screen margin
            (e.g. "tl" = top-left corner, "r" = right edge at the slot's cy) so the photo grows
            from there into open space; without it the photo grows around (cx, cy)."""
            im, sl = srcs[shots[i]["image"]], slots[shots[i]["slot"]]
            w = hgt * im.width / im.height
            a = sl.get("anchor", "")
            x = margin if "l" in a else (W - margin - w if "r" in a else sl["cx"] - w / 2)
            y = margin if "t" in a else (H - margin - hgt if "b" in a else sl["cy"] - hgt / 2)
            return (int(round(x)), int(round(y)), int(round(w)), int(round(hgt)))

        for g in groups:
            hs = {i: 60.0 for i in g}
            changed = True
            while changed:
                changed = False
                for i in g:
                    b = box_of(i, hs[i] * 1.01)
                    sl = slots[shots[i]["slot"]]
                    if b[2] > sl.get("max_w", W) or b[3] > sl.get("max_h", H):
                        continue
                    if b[0] < margin or b[1] < margin or b[0] + b[2] > W - margin or b[1] + b[3] > H - margin:
                        continue
                    if any(boxes_overlap(b, box_of(j, hs[j]), gap) for j in g if j != i):
                        continue
                    hs[i] *= 1.01
                    changed = True
            for i in g:
                grown[i] = box_of(i, hs[i])
    for i, sh in enumerate(shots):
        if sh.get("clear"):
            layers = []
        else:
            key, slot = sh["image"], slots.get(sh["slot"], {})
            im = srcs[key]
            if "box" in sh:
                box = tuple(sh["box"])
                w, h = box[2], box[3]
                if abs(w / h - im.width / im.height) > 0.01:
                    errs.append(f"step {i + 1}: {key} box changes the photo's shape")
            elif i in grown:
                box = grown[i]
                w, h = box[2], box[3]
            else:
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
    for n, boxes in enumerate(cfg.get("_group_boxes", [])):
        bad = spacing_check(boxes, gap)
        if bad:
            errs.append(f"group {n + 1}: uneven spacing {bad}")
    if errs:
        sys.exit("layout errors:\n  " + "\n  ".join(errs))
    used = [s["image"] for s in shots if not s.get("clear")]
    if len(set(used)) != len(used) and not cfg.get("allow_repeats"):
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
           f"layout: no photo overlaps or comes within {gap}px of another, none cropped" + ("" if cfg.get("allow_repeats") else ", none repeated")]
    if "_coverage" in cfg:
        rep.append(f"spacing: exactly {gap}px between every pair of neighbouring photos -> PASS")
        rep.append("screen filled per group: " + ", ".join(f"{c:.0%}" for c in cfg["_coverage"]))
    rep.append("")
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
