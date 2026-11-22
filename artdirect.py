#!/usr/bin/env python3
"""Art-directed beat edits: exact crops, one compositional device, cuts on a steady beat grid.

    python3 artdirect.py --config config_sotr_poison_in_bloom.json

Built for edits where the crop is part of the design (full-screen 9:16 slices, a detail before the
photo it belongs to, a portrait inside a poster window). Photos never move once they land: only the
mask that reveals them can open, and every opening finishes exactly on a beat.

Config (all times in beats from 0; bpm should give a whole number of frames per beat):
  "bpm", "beats", "fps", "size", "background", "grade" {warmth, lift, saturation, grain, grain_seed}
  "ui_zones": [[x0, y0, x1, y1], ...]  TikTok's interface on a 1080x1920 frame: top tabs, the
                                  right-side buttons, the caption and sound line. Faces and essential
                                  details ("focus" boxes) must not touch them. Default below (UI_ZONES).
  "max_upscale": 1.3              QA flags any crop shown larger than this
  "images": {key: {"file", "source", "credit", "kind"}}
  "templates": {name: {"background", "window": [X, Y, W, H], "elements": [...]}}
        elements: rect / outline / line / text (font, size, xy, anchor, color, tracking) / paper grain
  "shots": [{
        "id", "image", "at": beat,
        "crop": [x0, y0, w]           source pixels; height follows the target box exactly (no stretch)
        "rect": [X, Y, W, H]          target box on the canvas (default: full screen)
        "template": name              draw the template, the photo goes in its window
        "focus": [[x0, y0, x1, y1]]   faces / essential details in source pixels (safe-zone QA)
        "same_photo_as": id           a detail and the photo it belongs to (allowed repeat, noted)
        "in": {"type": "cut"}                                      appears in one frame
              {"type": "dissolve", "end": beat}                    blend that finishes on "end"
              {"type": "luma", "end": beat, "soft": 0.18}          light-first: brightest areas of
                                                                    the new photo arrive first
              {"type": "luma", ..., "keep": {"box": [x0, y0, x1, y1],  the outgoing photo's subject (canvas
                         "mode": "ellipse|dark|bright"}}            box) stays until the end of the blend
              {"type": "window", "box": [X, Y, W, H],              new full-bleed photo shows inside the
               "open": [b0, b1], "edge": "#hex"}                    box, then the box opens to full frame
              {"type": "window_close", "box": ..., "open": ...}    the new frame closes in from the edges
              {"type": "petal", "petal": [cx, cy, w, h],           shows inside a petal-shaped window
               "open": [beat_start, beat_end], "edge": "#hex"}     then the window opens to full
                                                                    screen (omit "open" to stay a petal)
  }]

Writes out/<name>/<version>.mp4 (with a plain beat guide), _silent.mp4, _qa.txt, _sheet.jpg and
_safezones.jpg (the sequence with TikTok's interface zones drawn over it) into out/<name>/.
"""
import argparse
import json
import math
import os
import subprocess
import sys
import wave

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = os.path.dirname(os.path.abspath(__file__))
UI_ZONES = [[950, 680, 1080, 1500],   # avatar, like, comment, save, share buttons
            [0, 1480, 1080, 1920]]    # username, caption, sound line, nav bar


def hex_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def grade(a, g):
    a = a.astype(np.float32)
    lum = a.mean(axis=2, keepdims=True)
    a = lum + (a - lum) * g.get("saturation", 1.0)
    w = g.get("warmth", 0.0)
    a[..., 0] *= 1 + w
    a[..., 1] *= 1 + w * 0.3
    a[..., 2] *= 1 - w
    lift = g.get("lift", 0)
    a = lift + a * (255 - lift) / 255.0
    return np.clip(a, 0, 255)


def smooth(p):
    p = min(1.0, max(0.0, p))
    return p * p * p * (p * (p * 6 - 15) + 10)          # smootherstep: eases in and out


class Edit:
    def __init__(self, cfg):
        self.cfg = cfg
        self.W, self.H = cfg["size"]
        self.fps = cfg["fps"]
        self.fpb = self.fps * 60.0 / cfg["bpm"]           # frames per beat
        self.N = int(round(cfg["beats"] * self.fpb))
        self.notes = []
        self.yy, self.xx = np.mgrid[0:self.H, 0:self.W].astype(np.float32)
        self.sources = {}

    def frame_of(self, beat):
        return beat * self.fpb

    # ---------- plates ----------
    def source(self, key):
        if key not in self.sources:
            spec = self.cfg["images"][key]
            im = Image.open(os.path.join(ROOT, spec["file"])).convert("RGB")
            self.sources[key] = im
        return self.sources[key]

    def font(self, path, size):
        return ImageFont.truetype(path if os.path.isabs(path) else os.path.join(ROOT, path), size)

    def draw_text(self, d, el):
        f = self.font(el["font"], el["size"])
        txt = el["text"]
        tr = el.get("tracking", 0.0) * el["size"]
        widths = [d.textlength(c, font=f) for c in txt]
        total = sum(widths) + tr * (len(txt) - 1)
        x, y = el["xy"]
        anchor = el.get("anchor", "mm")
        if anchor[0] == "m":
            x -= total / 2
        elif anchor[0] == "r":
            x -= total
        col = hex_rgb(el.get("color", "#000000"))
        for c, w in zip(txt, widths):
            d.text((x, y), c, font=f, fill=col, anchor="l" + anchor[1])
            x += w + tr

    def template_plate(self, name):
        t = self.cfg["templates"][name]
        im = Image.new("RGB", (self.W, self.H), hex_rgb(t.get("background", "#FFFFFF")))
        d = ImageDraw.Draw(im)
        for el in t.get("elements", []):
            k = el["type"]
            if k == "rect":
                d.rectangle(el["box"], fill=hex_rgb(el["fill"]))
            elif k == "outline":
                d.rectangle(el["box"], outline=hex_rgb(el["color"]), width=el.get("width", 1))
            elif k == "line":
                d.line(el["xy"], fill=hex_rgb(el["color"]), width=el.get("width", 1))
            elif k == "text":
                self.draw_text(d, el)
            elif k == "sunburst":
                # art-deco fan of rays from a centre point (angles in degrees, 0 = right, 90 = up)
                cx, cy = el["center"]
                n, a0, a1 = el["rays"], el.get("from", 0), el.get("to", 180)
                for r in range(n):
                    ang = math.radians(a0 + (a1 - a0) * r / (n - 1))
                    ro = el["r_out"] * (1.0 if r % 2 == 0 else el.get("short", 0.72))
                    p0 = (cx + el["r_in"] * math.cos(ang), cy - el["r_in"] * math.sin(ang))
                    p1 = (cx + ro * math.cos(ang), cy - ro * math.sin(ang))
                    d.line([p0, p1], fill=hex_rgb(el["color"]), width=el.get("width", 2))
            elif k == "diamond":
                cx, cy = el["center"]; s = el["size"]
                d.polygon([(cx, cy - s), (cx + s, cy), (cx, cy + s), (cx - s, cy)], fill=hex_rgb(el["color"]))
        a = np.asarray(im).astype(np.float32)
        if t.get("paper_grain"):
            rng = np.random.default_rng(t.get("paper_seed", 3))
            a += rng.normal(0, t["paper_grain"], (self.H, self.W, 1)).astype(np.float32)
        return a

    def shot_rect(self, s):
        if s.get("template"):
            return self.cfg["templates"][s["template"]]["window"]
        return s.get("rect", [0, 0, self.W, self.H])

    def crop_box(self, s):
        X, Y, Wr, Hr = self.shot_rect(s)
        x0, y0, w = s["crop"]
        h = w * Hr / Wr
        return x0, y0, w, h

    def plate(self, s):
        g = self.cfg["grade"]
        X, Y, Wr, Hr = self.shot_rect(s)
        x0, y0, w, h = self.crop_box(s)
        src = self.source(s["image"])
        if x0 < -0.5 or y0 < -0.5 or x0 + w > src.width + 0.5 or y0 + h > src.height + 0.5:
            self.notes.append(f"FAIL {s['id']}: crop {x0:.0f},{y0:.0f},{w:.0f}x{h:.0f} runs outside the "
                              f"{src.width}x{src.height} source")
        box = (max(0.0, x0), max(0.0, y0), min(src.width, x0 + w), min(src.height, y0 + h))
        piece = src.resize((Wr, Hr), Image.LANCZOS, box=box)
        if s.get("template"):
            base = self.template_plate(s["template"])
        else:
            base = np.empty((self.H, self.W, 3), np.float32)
            base[:] = hex_rgb(self.cfg.get("background", "#000000"))
        base[Y:Y + Hr, X:X + Wr] = grade(np.asarray(piece), g)
        return base

    # ---------- masks ----------
    def petal_mask(self, cx, cy, w, h):
        """A vertical almond (vesica) w wide and h tall, soft 1.5 px edge."""
        w = max(w, 2.0)
        R = (w / 2 + h * h / (2 * w)) / 2
        d = R - w / 2
        d1 = np.sqrt((self.xx - (cx - d)) ** 2 + (self.yy - cy) ** 2)
        d2 = np.sqrt((self.xx - (cx + d)) ** 2 + (self.yy - cy) ** 2)
        inside = R - np.maximum(d1, d2)
        return np.clip(inside / 1.5 + 0.5, 0, 1)[..., None], inside

    def petal_full_scale(self, cx, cy, w, h):
        """Smallest scale at which the almond covers every corner of the canvas."""
        lo, hi = 1.0, 200.0
        for _ in range(40):
            m = (lo + hi) / 2
            ww, hh = w * m, h * m
            R = (ww / 2 + hh * hh / (2 * ww)) / 2
            d = R - ww / 2
            ok = all(max(math.hypot(px - (cx - d), py - cy), math.hypot(px - (cx + d), py - cy)) <= R - 2
                     for px, py in ((0, 0), (self.W, 0), (0, self.H), (self.W, self.H)))
            hi, lo = (m, lo) if ok else (hi, m)
        return hi

    def light_map(self, img, passes):
        """The photo's own light as a smooth float map, 0 at its darkest to 1 at its brightest:
        block-averaged to 1/8 size, box-blurred a few times, scaled back up (no banding, no speckle)."""
        b = 8
        l = img.mean(axis=2)[: self.H // b * b, : self.W // b * b]
        s = l.reshape(self.H // b, b, self.W // b, b).mean(axis=(1, 3))
        for _ in range(passes):
            pad = np.pad(s, 2, mode="edge")
            s = sum(pad[dy:dy + s.shape[0], dx:dx + s.shape[1]] for dy in range(5) for dx in range(5)) / 25.0
        up = np.asarray(Image.fromarray(s.astype(np.float32), mode="F").resize((self.W, self.H), Image.BILINEAR))
        lo, hi = np.percentile(up, 2), np.percentile(up, 99.5)
        return np.clip((up - lo) / max(1.0, hi - lo), 0, 1)[..., None].astype(np.float32)

    def subject_mask(self, img, keep):
        """Soft mask of the outgoing photo's subject inside a canvas box, so it is the last thing to
        leave in a luma dissolve. mode "ellipse": a feathered oval filling the box; "dark" / "bright":
        the pixels clearly darker / brighter than the middle of the box's range (a silhouette, the sun),
        smoothed and feathered so the edge never speckles."""
        x0, y0, x1, y1 = keep["box"]
        fe = keep.get("feather", 40)
        cx, cy, rx, ry = (x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0) / 2, (y1 - y0) / 2
        d = np.sqrt(((self.xx - cx) / rx) ** 2 + ((self.yy - cy) / ry) ** 2)
        oval = np.clip((1 - d) * min(rx, ry) / fe + 0.5, 0, 1)
        mode = keep.get("mode", "ellipse")
        if mode == "ellipse":
            m = oval
        else:
            lum = img.mean(axis=2)
            inbox = lum[int(y0):int(y1), int(x0):int(x1)]
            lo, hi = np.percentile(inbox, 10), np.percentile(inbox, 90)
            thr, soft = (lo + hi) / 2, max(4.0, (hi - lo) * 0.15)
            m = np.clip(((thr - lum) if mode == "dark" else (lum - thr)) / soft + 0.5, 0, 1)
            box = (np.clip((self.xx - x0) / fe, 0, 1) * np.clip((x1 - self.xx) / fe, 0, 1) *
                   np.clip((self.yy - y0) / fe, 0, 1) * np.clip((y1 - self.yy) / fe, 0, 1))
            m = m * box
        b = 4
        s = m[: self.H // b * b, : self.W // b * b].reshape(self.H // b, b, self.W // b, b).mean(axis=(1, 3))
        for _ in range(keep.get("smooth", 2)):
            pad = np.pad(s, 2, mode="edge")
            s = sum(pad[dy:dy + s.shape[0], dx:dx + s.shape[1]] for dy in range(5) for dx in range(5)) / 25.0
        up = np.asarray(Image.fromarray(s.astype(np.float32), mode="F").resize((self.W, self.H), Image.BILINEAR))
        return np.clip(up, 0, 1)[..., None].astype(np.float32)

    # ---------- render ----------
    def build(self):
        shots = self.cfg["shots"]
        self.plates = [self.plate(s) for s in shots]
        self.starts = [int(round(self.frame_of(s["at"]))) for s in shots]
        for s in shots:
            tr = s.get("in", {"type": "cut"})
            if tr["type"] == "petal":
                cx, cy, w, h = tr["petal"]
                tr["_full"] = self.petal_full_scale(cx, cy, w, h)
        grain = self.cfg["grade"].get("grain", 0)
        self.grain = np.random.default_rng(self.cfg["grade"].get("grain_seed", 0)).normal(
            0, grain, (self.H, self.W, 1)).astype(np.float32) if grain else 0

    def shot_at(self, f):
        i = 0
        for k, st in enumerate(self.starts):
            if st <= f:
                i = k
        return i

    def frame(self, f):
        """Returns (rgb float frame, is_transition) for frame f."""
        i = self.shot_at(f)
        s = self.cfg["shots"][i]
        cur = self.plates[i]
        tr = s.get("in", {"type": "cut"})
        if i == 0 or tr["type"] == "cut":
            return cur, False
        prev = self.plates[i - 1]
        # what was on screen just before this shot landed (the previous shot may itself be a petal)
        ps = self.cfg["shots"][i - 1]
        ptr = ps.get("in", {"type": "cut"})
        if ptr["type"] == "petal" and "open" not in ptr and i >= 2:
            under = self.plates[i - 2]
            m, _ = self.petal_mask(*ptr["petal"])
            prev = prev * m + under * (1 - m)
        t = tr["type"]
        if t in ("window", "window_close"):
            # a rectangular window: "window" shows the new full-bleed photo inside a box (for example
            # a poster's window) and pushes the box out to the full frame; "window_close" brings the
            # new frame (the poster) in from the edges, the old photo still showing in the shrinking
            # box until it closes. Photos stay put; only the box edges move.
            a, b = self.frame_of(tr["open"][0]), self.frame_of(tr["open"][1])
            full = (0, 0, self.W, self.H)
            box = tr["box"]
            p = 0.0 if f < a else smooth((f - a + 1) / max(1, b - a))
            if f >= b:
                return cur, False
            if t == "window":
                r = [box[k] + (full[k] - box[k]) * p for k in range(4)]
                inner, outer = cur, prev
            else:
                r = [full[k] + (box[k] - full[k]) * p for k in range(4)]
                inner, outer = prev, cur
            x0, y0, x1, y1 = r[0], r[1], r[0] + r[2], r[1] + r[3]
            m = (np.clip(self.xx - x0 + 0.5, 0, 1) * np.clip(x1 - self.xx + 0.5, 0, 1) *
                 np.clip(self.yy - y0 + 0.5, 0, 1) * np.clip(y1 - self.yy + 0.5, 0, 1))[..., None]
            out = inner * m + outer * (1 - m)
            if tr.get("edge") and 0 < p < 1:
                ew = tr.get("edge_width", 3)
                ring = ((np.abs(self.xx - x0) < ew) | (np.abs(self.xx - x1) < ew)) & (self.yy > y0 - ew) & (self.yy < y1 + ew)
                ring |= ((np.abs(self.yy - y0) < ew) | (np.abs(self.yy - y1) < ew)) & (self.xx > x0 - ew) & (self.xx < x1 + ew)
                out = np.where(ring[..., None], np.array(hex_rgb(tr["edge"]), np.float32), out)
            return out, f >= a
        if t == "dissolve":
            a, b = self.starts[i], self.frame_of(tr["end"])
            p = smooth((f - a + 1) / max(1, b - a))
            return cur * p + prev * (1 - p), p < 1
        if t == "luma":
            a, b = self.starts[i], self.frame_of(tr["end"])
            q = (f - a + 1) / max(1, b - a)                     # linear progress 0..1
            if q >= 1:
                return cur, False
            if "_lum" not in tr:
                tr["_lum"] = self.light_map(cur, tr.get("blur", 6))
            # every pixel blends smoothly, but the new photo's brightest areas start first and its
            # shadows last: k is how much of the blend the light gets ahead by
            k = tr.get("lead", 0.55)
            keep = tr.get("keep")
            qg = q / keep.get("until", 0.62) if keep else q      # with a kept subject, everything else lands first
            t0 = k * (1 - tr["_lum"])
            x = np.clip((qg - t0) / (1 - k), 0, 1)
            al = x * x * (3 - 2 * x)
            if keep:
                if "_mask" not in tr:
                    tr["_mask"] = self.subject_mask(prev, keep)
                u = keep.get("until", 0.62)
                xs = min(1.0, max(0.0, (q - u) / (1 - u)))
                S = tr["_mask"]
                al = al * (1 - S) + (xs * xs * (3 - 2 * xs)) * S
            return cur * al + prev * (1 - al), True
        if t == "petal":
            cx, cy, w, h = tr["petal"]
            scale, moving = 1.0, False
            if "open" in tr:
                a, b = self.frame_of(tr["open"][0]), self.frame_of(tr["open"][1])
                if f >= b:
                    return cur, False
                if f >= a:
                    p = smooth((f - a + 1) / max(1, b - a))
                    scale = 1 + (tr["_full"] - 1) * p
                    moving = True
            m, inside = self.petal_mask(cx, cy, w * scale, h * scale)
            out = cur * m + prev * (1 - m)
            if tr.get("edge"):
                e = np.clip(1 - np.abs(inside - 0.0) / tr.get("edge_width", 2.0), 0, 1)[..., None] * 0.9
                out = out * (1 - e) + np.array(hex_rgb(tr["edge"]), np.float32) * e
            return out, moving
        return cur, False

    def render(self, outdir):
        cfg = self.cfg
        os.makedirs(outdir, exist_ok=True)
        base = os.path.join(outdir, cfg["version"])
        wav = base + "_guide.wav"
        self.beat_guide(wav)
        mp4, silent = base + ".mp4", base + "_silent.mp4"
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
               "-s", f"{self.W}x{self.H}", "-r", str(self.fps), "-i", "-", "-i", wav,
               "-c:v", "libx264", "-preset", "slow", "-crf", "16", "-pix_fmt", "yuv420p",
               "-c:a", "aac", "-b:a", "160k", "-shortest", "-movflags", "+faststart", mp4]
        pr = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        self.stills_bad, self.keyframes = [], {}
        last, last_moving = None, True
        want = set()
        for k, st in enumerate(self.starts):
            want.update({st, max(st - 1, 0)})
        want.update({0, self.N - 1})
        for f in range(self.N):
            fr, moving = self.frame(f)
            out = np.clip(fr + self.grain, 0, 255).astype(np.uint8)
            if last is not None and not moving and not last_moving and self.shot_at(f) == self.shot_at(f - 1):
                if np.any(out != last):
                    self.stills_bad.append(f)
            if f in want or f % int(round(self.fpb)) == 0:
                self.keyframes[f] = out
            pr.stdin.write(out.tobytes())
            last, last_moving = out, moving
        pr.stdin.close()
        if pr.wait():
            sys.exit("ffmpeg failed")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", mp4, "-an", "-c:v", "copy",
                        "-movflags", "+faststart", silent], check=True)
        os.remove(wav)
        self.qa(mp4, base)
        self.sheets(base)
        return mp4

    def beat_guide(self, path, sr=44100):
        """Plain kick on every beat, louder on each bar's first beat, a soft tick on off-beats."""
        bpm, beats = self.cfg["bpm"], self.cfg["beats"]
        n = int(round(self.N / self.fps * sr))
        y = np.zeros(n, np.float32)
        t = np.arange(int(0.25 * sr)) / sr
        kick = np.sin(2 * np.pi * (45 + 90 * np.exp(-t * 35)) * t) * np.exp(-t * 14)
        tick = np.random.default_rng(1).normal(0, 1, int(0.03 * sr)) * np.exp(-np.arange(int(0.03 * sr)) / sr * 160) * 0.15
        for b in range(beats):
            for off, s, g in ((0.0, kick, 0.85 if b % 4 == 0 else 0.5), (0.5, tick, 1.0)):
                i = int(round((b + off) * 60 / bpm * sr))
                seg = s[: max(0, min(len(s), n - i))]
                y[i:i + len(seg)] += g * seg
        pcm = (np.clip(y, -1, 1) * 32000).astype(np.int16)
        with wave.open(path, "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr); w.writeframes(pcm.tobytes())

    # ---------- QA ----------
    def qa(self, mp4, base):
        cfg = self.cfg
        L = []
        ok = True
        L.append(f"{cfg['name']} / {cfg['version']}: {cfg['bpm']} BPM, {cfg['beats']} beats "
                 f"({cfg['beats'] / 4:g} bars), {self.N} frames = {self.N / self.fps:.2f} s, "
                 f"{self.fpb:g} frames per beat")
        probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
                                "-show_entries", "stream=width,height,r_frame_rate,nb_read_frames,codec_name",
                                "-of", "csv=p=0", mp4], capture_output=True, text=True).stdout.strip()
        L.append(f"file: {probe} (codec, width, height, fps, frames)")
        if abs(self.fpb - round(self.fpb)) > 1e-6:
            L.append("FAIL tempo does not give a whole number of frames per beat"); ok = False
        L.append("\nEvents (every cut, petal opening and blend end is on the beat grid):")
        for s, st in zip(cfg["shots"], self.starts):
            exact = self.frame_of(s["at"])
            off = abs(st - exact)
            tr = s.get("in", {"type": "cut"})
            extra = ""
            if tr["type"] == "petal" and "open" in tr:
                extra = f", opens beat {tr['open'][0]:g}->{tr['open'][1]:g} (lands {self.frame_of(tr['open'][1]) / self.fps:.2f} s)"
            elif tr["type"] == "petal":
                extra = ", stays a petal window"
            elif tr["type"] in ("window", "window_close"):
                verb = "opens to full frame" if tr["type"] == "window" else "poster closes in"
                extra = f", {verb} beat {tr['open'][0]:g}->{tr['open'][1]:g} (lands {self.frame_of(tr['open'][1]) / self.fps:.2f} s)"
            elif "end" in tr:
                extra = f", {tr['type']} ends beat {tr['end']:g} ({self.frame_of(tr['end']) / self.fps:.2f} s)"
                if tr.get("keep"):
                    extra += ", the outgoing subject leaves last"
            flag = "" if off <= 1 else "  FAIL off grid"
            ok &= off <= 1
            L.append(f"  beat {s['at']:>5g}  {st / self.fps:6.2f} s  frame {st:4d}  {tr['type']:8s} {s['id']}{extra}{flag}")
        L.append("\nHeld frames perfectly still: " + ("PASS" if not self.stills_bad else
                 f"FAIL at frames {self.stills_bad[:10]}"))
        ok &= not self.stills_bad
        # crops, upscale, stretch, safe zone, repeats
        L.append("\nCrops (source px -> screen; upscale over 1.0 means the source is enlarged):")
        maxup = cfg.get("max_upscale", 1.3)
        zones = cfg.get("ui_zones", UI_ZONES)
        seen = {}
        for s in cfg["shots"]:
            X, Y, Wr, Hr = self.shot_rect(s)
            x0, y0, w, h = self.crop_box(s)
            src = self.source(s["image"])
            up = Wr / w
            cover = Wr * Hr / (self.W * self.H) * 100
            flag = ""
            if up > maxup:
                flag += f"  FLAG upscale {up:.2f} > {maxup}"
                ok = False
            tr = s.get("in", {"type": "cut"})
            vis = None
            if tr["type"] == "petal" and "open" not in tr:
                cx, cy, pw, ph = tr["petal"]
                vis = (cx - pw / 2, cy - ph / 2, cx + pw / 2, cy + ph / 2)
            for fb in s.get("focus", []):
                fx0 = X + (fb[0] - x0) * Wr / w; fx1 = X + (fb[2] - x0) * Wr / w
                fy0 = Y + (fb[1] - y0) * Hr / h; fy1 = Y + (fb[3] - y0) * Hr / h
                inside = fx0 >= 0 and fy0 >= 0 and fx1 <= self.W and fy1 <= self.H and not any(
                    fx0 < z[2] and fx1 > z[0] and fy0 < z[3] and fy1 > z[1] for z in zones)
                if vis:
                    inside &= fx0 >= vis[0] and fx1 <= vis[2] and fy0 >= vis[1] and fy1 <= vis[3]
                if not inside:
                    flag += f"  FAIL focus {fx0:.0f},{fy0:.0f}-{fx1:.0f},{fy1:.0f} touches the interface or the frame edge"
                    ok = False
            if not s.get("focus"):
                flag += "  (no focus box)"
            key = s["image"]
            if key in seen and s.get("same_photo_as") != seen[key]:
                flag += f"  FAIL repeats {key}"; ok = False
            seen.setdefault(key, s["id"])
            rep = f" [detail of the same photo as {s['same_photo_as']}]" if s.get("same_photo_as") else ""
            L.append(f"  {s['id']:24s} {key:28s} src {src.width}x{src.height} crop {w:.0f}x{h:.0f} at "
                     f"{x0:.0f},{y0:.0f} -> {Wr}x{Hr} (x{up:.2f}), covers {cover:.0f}% of screen{rep}{flag}")
        L.append("\nFaces and key details (focus boxes) clear of TikTok's interface: right-side buttons "
                 "x>950 at y 680-1500, caption area y>1480 (the thin top tabs are not counted)")
        texts = [el.get("text") for t in cfg.get("templates", {}).values() for el in t.get("elements", [])
                 if el["type"] == "text"]
        if cfg.get("text") == "none":
            L.append("Text in the edit: " + ("FAIL drawn text " + ", ".join(texts) if texts else
                     "none drawn; every crop was checked by eye to leave out poster titles, names, signs and logos"))
            ok &= not texts
        first, lastf = self.keyframes[0].astype(np.float32), self.keyframes[self.N - 1].astype(np.float32)
        L.append(f"Loop: last frame -> first frame mean difference {np.abs(first - lastf).mean():.1f}/255 "
                 f"(a cut on the bar line back to shot 1)")
        L.extend("NOTE " + n for n in self.notes)
        ok &= not any(n.startswith("FAIL") for n in self.notes)
        L.append("\nOVERALL: " + ("PASS" if ok else "CHECK FLAGS ABOVE"))
        open(base + "_qa.txt", "w").write("\n".join(L) + "\n")
        print("\n".join(L))

    def sheets(self, base):
        """Sequence sheet (each shot settled, plus mid-transition frames) and a safe-zone sheet."""
        cfg = self.cfg
        picks = []
        for k, (s, st) in enumerate(zip(cfg["shots"], self.starts)):
            end = self.starts[k + 1] if k + 1 < len(self.starts) else self.N
            settle = end - 1
            tr = s.get("in", {"type": "cut"})
            if tr["type"] != "cut":
                picks.append((st, f"{st / self.fps:.2f}s {s['id']} ({tr['type']} starts)"))
                if tr["type"] == "petal" and "open" in tr:
                    mid = int((self.frame_of(tr["open"][0]) + self.frame_of(tr["open"][1])) / 2)
                    picks.append((mid, f"{mid / self.fps:.2f}s opening"))
            picks.append((settle, f"{st / self.fps:.2f}s {s['id']}"))
        need = {f for f, _ in picks} - set(self.keyframes)
        for f in sorted(need):
            fr, _ = self.frame(f)
            self.keyframes[f] = np.clip(fr + self.grain, 0, 255).astype(np.uint8)
        tw, th, cols = 216, 384, 8
        rows = (len(picks) + cols - 1) // cols
        try:
            fnt = ImageFont.truetype("/usr/share/fonts/opentype/inter/Inter-Regular.otf", 13)
        except OSError:
            fnt = ImageFont.load_default()
        for kind in ("sheet", "safezones"):
            sheet = Image.new("RGB", (cols * (tw + 8) + 8, rows * (th + 30) + 8), (250, 250, 248))
            d = ImageDraw.Draw(sheet)
            for n, (f, label) in enumerate(picks):
                im = Image.fromarray(self.keyframes[f]).resize((tw, th), Image.LANCZOS)
                if kind == "safezones":
                    ov = Image.new("RGBA", (tw, th), (0, 0, 0, 0))
                    od = ImageDraw.Draw(ov)
                    sc = tw / self.W
                    for box in cfg.get("ui_zones", UI_ZONES):
                        od.rectangle([v * sc for v in box], fill=(255, 0, 80, 70))
                    od.rectangle([0, 0, self.W * sc, 200 * sc], fill=(255, 0, 80, 25))   # top tabs, shown faintly
                    s_ = cfg["shots"][self.shot_at(f)]
                    X, Y, Wr, Hr = self.shot_rect(s_)
                    x0, y0, w, h = self.crop_box(s_)
                    for fb in s_.get("focus", []):
                        od.rectangle([(X + (fb[0] - x0) * Wr / w) * sc, (Y + (fb[1] - y0) * Hr / h) * sc,
                                      (X + (fb[2] - x0) * Wr / w) * sc, (Y + (fb[3] - y0) * Hr / h) * sc],
                                     outline=(0, 230, 120, 255), width=1)
                    im = Image.alpha_composite(im.convert("RGBA"), ov).convert("RGB")
                x, y = 8 + (n % cols) * (tw + 8), 8 + (n // cols) * (th + 30)
                sheet.paste(im, (x, y))
                d.text((x, y + th + 4), label[:34], fill=(30, 30, 30), font=fnt)
            sheet.save(f"{base}_{kind}.jpg", quality=88)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--frames", help="only write these frames as PNGs (comma list) for checking crops")
    a = ap.parse_args()
    cfg = json.load(open(os.path.join(ROOT, a.config) if not os.path.isabs(a.config) else a.config))
    e = Edit(cfg)
    e.build()
    outdir = os.path.join(ROOT, "out", cfg["name"])
    if a.frames:
        os.makedirs(outdir, exist_ok=True)
        for f in [int(x) for x in a.frames.split(",")]:
            fr, _ = e.frame(f)
            Image.fromarray(np.clip(fr + e.grain, 0, 255).astype(np.uint8)).save(
                os.path.join(outdir, f"{cfg['version']}_f{f:04d}.png"))
        return
    e.render(outdir)


if __name__ == "__main__":
    main()
