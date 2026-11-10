#!/usr/bin/env python3
"""Beat-synced aesthetic mood board edits for TikTok.

    python3 moodboard.py                 # all versions in config.json
    python3 moodboard.py --config other.json --only C_pov_loop

Pipeline: audio analysis -> cut grid -> curation sheet -> render (still images,
opacity-only dissolves that finish on the beat) -> QA on the rendered file.
Outputs go to out/<name>/.
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = os.path.dirname(os.path.abspath(__file__))


# --------------------------------------------------------------------------- audio
def analyze_audio(cfg, units_needed_beats_fn):
    """Return dict with beat times (absolute, in song), tempo, downbeat phase,
    window start beat index, and whether this is a placeholder grid."""
    song = os.path.join(ROOT, cfg["song"]) if cfg.get("song") else None
    if not song or not os.path.exists(song):
        bpm = float(cfg["fallback_bpm"])
        beats = np.arange(0, 256) * 60.0 / bpm
        return dict(placeholder=True, tempo=bpm, beats=beats, phase=0, song=None,
                    onset_t=None, onset=None)

    import librosa
    y, sr = librosa.load(song, sr=22050, mono=True)
    hop = 512
    onset = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    tempo, beat_frames = librosa.beat.beat_track(onset_envelope=onset, sr=sr, hop_length=hop, trim=False)
    tempo = float(np.atleast_1d(tempo)[0])
    beats = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop)
    onset_t = librosa.times_like(onset, sr=sr, hop_length=hop)
    # downbeat estimate: the beat phase (of 4) with the most low-end (kick) energy
    mel = librosa.feature.melspectrogram(y=y, sr=sr, hop_length=hop, n_mels=64)
    low = mel[:12].sum(0)  # raw kick/bass-band energy
    peak = np.array([low[b:b + 4].max() if b < len(low) else 0 for b in beat_frames])
    phase = int(np.argmax([peak[p::4].mean() for p in range(4)]))
    rms = librosa.feature.rms(y=y, hop_length=hop)[0]
    return dict(placeholder=False, tempo=tempo, beats=beats, phase=phase, song=song,
                onset_t=onset_t, onset=onset, rms=rms, duration=len(y) / sr)


def beat_time(beats, pos):
    """Absolute time of fractional beat position (interpolated, extrapolated at the ends)."""
    i = int(np.floor(pos))
    frac = pos - i
    if i + 1 < len(beats) and i >= 0:
        return beats[i] + frac * (beats[i + 1] - beats[i])
    period = np.median(np.diff(beats))
    return beats[-1] + (pos - (len(beats) - 1)) * period


def choose_window(cfg, au, total_beats):
    """Pick start beat index (a downbeat) for a window of total_beats beats."""
    beats = au["beats"]
    downbeats = list(range(au["phase"], len(beats) - int(np.ceil(total_beats)) - 1, 4))
    if au["placeholder"]:
        return 0, "placeholder grid (no song file): starts at 0"
    if cfg.get("song_start") is not None:
        s = min(downbeats, key=lambda i: abs(beats[i] - cfg["song_start"]))
        return s, f"forced near song_start={cfg['song_start']}s -> nearest downbeat {beats[s]:.2f}s"
    # score = loudness + onset activity, normalised, over the window
    hop_t = au["onset_t"][1] - au["onset_t"][0]
    rms, on = au["rms"] / (au["rms"].max() + 1e-9), au["onset"] / (au["onset"].max() + 1e-9)
    best, best_s = None, -1
    for s in downbeats:
        a, b = beats[s], beat_time(beats, s + total_beats)
        ia, ib = int(a / hop_t), int(b / hop_t)
        if ib >= len(rms):
            break
        score = rms[ia:ib].mean() + 0.5 * on[ia:ib].mean()
        if best is None or score > best:
            best, best_s = score, s
    return best_s, f"auto: loudest/busiest {total_beats:g}-beat window starting on a downbeat (score {best:.3f})"


# --------------------------------------------------------------------------- images
def hex_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def load_source(cfg, key):
    meta = cfg["images"].get(key, {})
    path = None
    for ext in (".jpg", ".jpeg", ".png", ".webp"):
        p = os.path.join(ROOT, cfg["images_dir"], key + ext)
        if os.path.exists(p):
            path = p
            break
    im = Image.open(path).convert("RGB")
    if meta.get("crop"):
        im = im.crop(tuple(meta["crop"]))
    return im, meta


def cover(im, w, h, focus=(0.5, 0.5)):
    """Scale to cover w x h, crop around focus point. Done once per image."""
    s = max(w / im.width, h / im.height)
    nw, nh = int(round(im.width * s)), int(round(im.height * s))
    im = im.resize((nw, nh), Image.LANCZOS)
    x = int(round((nw - w) * focus[0]))
    y = int(round((nh - h) * focus[1]))
    return im.crop((x, y, x + w, y + h))


def grade(arr, g):
    """Light, identical grade on every image: warmth, lifted blacks, slight desat."""
    a = arr.astype(np.float32)
    lum = a.mean(axis=2, keepdims=True)
    a = lum + (a - lum) * g["saturation"]
    a[..., 0] *= 1 + g["warmth"]
    a[..., 1] *= 1 + g["warmth"] * 0.3
    a[..., 2] *= 1 - g["warmth"]
    a = g["lift"] + a * (255 - g["lift"]) / 255.0
    return np.clip(a, 0, 255)


def build_plates(cfg, ver, keys):
    """Pre-size and grade every image exactly once -> full canvas float arrays."""
    W, H = cfg["size"]
    g = cfg["grade"]
    plates = {}
    if ver["preset"] == "A":
        bg = np.empty((H, W, 3), np.float32)
        bg[:] = hex_rgb(ver.get("background", "#F7F4EE"))
        bw = int(round(W * ver.get("frame_width", 0.8)))
        bh = int(round(bw * 4 / 3))
        bx, by = (W - bw) // 2, (H - bh) // 2
        if ver.get("shadow"):
            sh = Image.new("L", (W, H), 0)
            ImageDraw.Draw(sh).rectangle((bx + 6, by + 14, bx + bw + 6, by + bh + 14), fill=60)
            sh = np.asarray(sh.filter(ImageFilter.GaussianBlur(22)), np.float32)[..., None] / 255.0
            bg = bg * (1 - sh * 0.55)
    for k in keys:
        im, meta = load_source(cfg, k)
        focus = tuple(meta.get("focus", (0.5, 0.5)))
        if ver["preset"] == "A":
            photo = grade(np.asarray(cover(im, bw, bh, focus)), g)
            canvas = bg.copy()
            canvas[by:by + bh, bx:bx + bw] = photo
        else:
            canvas = grade(np.asarray(cover(im, W, H, focus)), g)
        plates[k] = canvas
    rng = np.random.default_rng(g.get("grain_seed", 0))
    grain = rng.normal(0, g["grain"], (H, W, 1)).astype(np.float32)  # ONE static layer
    return plates, grain


# --------------------------------------------------------------------------- timeline
def build_timeline(cfg, ver, au):
    fps = cfg["fps"]
    seq = cfg["sequence"] * ver.get("passes", 1)
    scale = ver.get("beat_scale", 1)
    units = [s["beats"] * scale for s in seq]
    total_units = sum(units)
    period = 60.0 / au["tempo"]
    lo, hi = cfg["target_seconds"]
    # pick beat subdivision so the edit lands in 8-12 s, closest to 10 s, prefer 1 beat
    cands = [1, 0.5, 2, 0.25]
    fits = [s for s in cands if lo <= total_units * s * period <= hi]
    sub = min(fits, key=lambda s: (abs(total_units * s * period - 10), cands.index(s))) if fits else \
        min(cands, key=lambda s: abs(total_units * s * period - 10))
    total_beats = total_units * sub
    start_beat, why = choose_window(cfg, au, total_beats)
    beats = au["beats"]
    t0 = beat_time(beats, start_beat)
    t_end = beat_time(beats, start_beat + total_beats)
    N = int(round((t_end - t0) * fps))

    shots, pos = [], 0.0
    for s, u in zip(seq, units):
        t_ideal = beat_time(beats, start_beat + pos * sub) - t0  # absolute, never accumulated
        shots.append(dict(image=s["image"], units=u, hold=s["beats"] > 1,
                          t_ideal=t_ideal, start_frame=int(round(t_ideal * fps))))
        pos += u
    for i, sh in enumerate(shots):
        sh["end_frame"] = shots[i + 1]["start_frame"] if i + 1 < len(shots) else N
    # dissolve into shot i (finishes exactly on its start frame); i==0 handled as loop point
    for i, sh in enumerate(shots):
        prev = shots[i - 1]
        prev_len = prev["end_frame"] - prev["start_frame"]
        if sh["hold"] or i == 0:
            L = min(8, max(4, round(0.25 * fps)), max(4, prev_len // 2 + 1))
        else:
            L = int(np.clip(round(0.30 * prev_len), 4, 8))
        sh["dissolve"] = int(L)
    for sh in shots:
        sh["start"] = round(sh["start_frame"] / fps, 4)
        sh["end"] = round(sh["end_frame"] / fps, 4)
    info = dict(tempo=au["tempo"], subdivision_beats=sub, total_beats=total_beats,
                window_start=t0, window_end=t_end, frames=N, why=why,
                shot_seconds=sub * period, placeholder=au["placeholder"])
    return shots, info


def frame_mix(f, shots, N):
    """(index_a, index_b, alpha) for frame f. alpha = weight of b."""
    for i, sh in enumerate(shots):
        if sh["start_frame"] <= f < sh["end_frame"]:
            nxt = (i + 1) % len(shots)
            target = sh["end_frame"]  # next image fully in at this frame (N == loop to frame 0)
            L = shots[nxt]["dissolve"]
            k = f - (target - L)
            if k > 0:
                x = k / L
                return i, nxt, x * x * (3 - 2 * x)  # ease-in-out
            return i, i, 0.0
    raise ValueError(f)


# --------------------------------------------------------------------------- render
def ffmpeg_audio_args(cfg, au, info, click_path):
    fade = 0.04
    d = info["frames"] / cfg["fps"]
    if au["placeholder"]:
        return ["-i", click_path], f"[1:a]atrim=0:{d:.4f},asetpts=N/SR/TB[a]"
    return (["-i", au["song"]],
            f"[1:a]atrim={info['window_start']:.4f}:{info['window_start'] + d:.4f},asetpts=N/SR/TB,"
            f"afade=t=in:d={fade},afade=t=out:st={d - fade:.4f}:d={fade}[a]")


def make_click(path, cfg, au, info, shots):
    """Placeholder audio for previews: soft clicks on every grid beat, accent on downbeats."""
    import soundfile as sf
    sr = 44100
    d = info["frames"] / cfg["fps"]
    y = np.zeros(int(sr * d) + sr // 10, np.float32)
    period = 60 / au["tempo"]
    n = int(d / period + 1e-6)
    for b in range(n):
        t = b * period
        f0, amp = (1500, 0.5) if b % 4 == 0 else (1000, 0.25)
        tt = np.arange(int(0.03 * sr)) / sr
        click = amp * np.sin(2 * np.pi * f0 * tt) * np.exp(-tt * 140)
        i = int(t * sr)
        y[i:i + len(click)] += click
    sf.write(path, y, sr)


def render(cfg, ver_name, ver, au, outdir):
    W, H = cfg["size"]
    fps = cfg["fps"]
    shots, info = build_timeline(cfg, ver, au)
    keys = list(dict.fromkeys(s["image"] for s in shots))
    plates, grain = build_plates(cfg, ver, keys)
    N = info["frames"]
    tl = dict(version=ver_name, preset=ver["preset"], fps=fps, info=info,
              shots=[{k: s[k] for k in ("image", "start", "end", "start_frame", "end_frame",
                                        "dissolve", "hold", "t_ideal")} for s in shots])
    with open(os.path.join(outdir, f"{ver_name}_timeline.json"), "w") as fh:
        json.dump(tl, fh, indent=1, default=float)

    tag = "_PREVIEW-click" if au["placeholder"] else ""
    mp4 = os.path.join(outdir, f"{ver_name}{tag}.mp4")
    silent = os.path.join(outdir, f"{ver_name}_silent.mp4")
    click = os.path.join(outdir, "_click.wav")
    if au["placeholder"]:
        make_click(click, cfg, au, info, shots)
    a_in, a_filter = ffmpeg_audio_args(cfg, au, info, click)
    venc = ["-c:v", "libx264", "-profile:v", "high", "-pix_fmt", "yuv420p", "-crf", "18",
            "-preset", "veryfast", "-r", str(fps), "-movflags", "+faststart"]
    cmd = (["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{W}x{H}", "-r", str(fps), "-i", "-"] + a_in +
           ["-filter_complex", a_filter, "-map", "0:v", "-map", "[a]"] + venc +
           ["-c:a", "aac", "-b:a", "192k", "-t", f"{N / fps:.4f}", mp4])
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    cache = {}
    for f in range(N):
        a, b, x = frame_mix(f, shots, N)
        key = (a, b, round(x, 6))
        if key not in cache:
            A = plates[shots[a]["image"]]
            fr = A if x == 0 else A * (1 - x) + plates[shots[b]["image"]] * x
            cache = {key: np.clip(fr + grain, 0, 255).astype(np.uint8).tobytes()}
        p.stdin.write(cache[key])
    p.stdin.close()
    if p.wait():
        sys.exit("ffmpeg failed")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", mp4, "-an", "-c:v", "copy",
                    "-movflags", "+faststart", silent], check=True)
    if os.path.exists(click):
        os.remove(click)
    return mp4, shots, info, plates


# --------------------------------------------------------------------------- QA
def decode_small(mp4, w=270, h=480):
    raw = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", mp4, "-vf", f"scale={w}:{h}",
                          "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3).astype(np.float32)


def qa(cfg, mp4, shots, info, outdir, ver_name):
    fps = cfg["fps"]
    F = decode_small(mp4)
    N = len(F)
    rep = [f"QA {ver_name}: {N} frames decoded (expected {info['frames']})"]
    # 1. stillness: consecutive frames outside dissolves
    still_diffs = []
    for i, sh in enumerate(shots):
        nxt = shots[(i + 1) % len(shots)]
        for f in range(sh["start_frame"] + 1, sh["end_frame"] - nxt["dissolve"] + 1):
            if f < N:
                still_diffs.append(np.abs(F[f] - F[f - 1]).mean())
    still_diffs = np.array(still_diffs)
    rep.append(f"stillness: {len(still_diffs)} held-frame pairs, mean diff {still_diffs.mean():.3f}, "
               f"max {still_diffs.max():.3f} (0-255 scale; <1 = codec noise only) -> "
               f"{'PASS' if still_diffs.max() < 1.0 else 'FAIL'}")
    # 2. beat sync: first frame where the incoming image is fully in vs. ideal beat time
    offs = []
    for i, sh in enumerate(shots):
        if i == 0:
            continue
        ref = F[min(sh["end_frame"] - shots[(i + 1) % len(shots)]["dissolve"], N - 1)]
        lo = sh["start_frame"] - sh["dissolve"] - 2
        hit = next(f for f in range(max(lo, 0), sh["end_frame"]) if np.abs(F[f] - ref).mean() < 1.0)
        offs.append(hit - sh["t_ideal"] * fps)
    offs = np.array(offs)
    rep.append(f"beat sync: {len(offs)} cuts, mean |offset| {np.abs(offs).mean():.2f} frames, "
               f"worst {np.abs(offs).max():.2f} frames -> {'PASS' if np.abs(offs).max() <= 1 else 'FAIL'}")
    # 3. loop point: last frame should be almost the first image
    loop_gap = np.abs(F[-1] - F[0]).mean()
    rep.append(f"loop: last->first frame diff {loop_gap:.2f} (last frame is {1 - 1 / shots[0]['dissolve']:.0%}+ "
               f"into the dissolve back to image 1)")
    with open(os.path.join(outdir, f"{ver_name}_qa.txt"), "w") as fh:
        fh.write("\n".join(rep) + "\n")
    return rep, F


def strip(F, centers, path, span=(-6, 2), label="transition"):
    rows = []
    for c in centers:
        fr = [F[int(np.clip(c + k, 0, len(F) - 1))] for k in range(span[0], span[1] + 1)]
        rows.append(np.concatenate([np.pad(x, ((4, 4), (4, 4), (0, 0)), constant_values=255) for x in fr], 1))
    Image.fromarray(np.concatenate(rows, 0).astype(np.uint8)).save(path, quality=88)


# --------------------------------------------------------------------------- sheets
def palette(im, k=5, seed=0):
    a = np.asarray(im.convert("RGB").resize((64, 64))).reshape(-1, 3).astype(np.float32)
    rng = np.random.default_rng(seed)
    c = a[rng.choice(len(a), k, replace=False)]
    for _ in range(15):
        lab = np.argmin(((a[:, None] - c[None]) ** 2).sum(-1), 1)
        c = np.array([a[lab == j].mean(0) if (lab == j).any() else c[j] for j in range(k)])
    cnt = np.bincount(lab, minlength=k)
    return [tuple(int(v) for v in c[j]) for j in np.argsort(-cnt)], cnt[np.argsort(-cnt)] / cnt.sum()


def font(sz):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",):
        if os.path.exists(p):
            return ImageFont.truetype(p, sz)
    return ImageFont.load_default()


def curation_sheet(cfg, path):
    names = sorted(os.path.splitext(f)[0] for f in os.listdir(os.path.join(ROOT, cfg["images_dir"])))
    tw, th = 300, 400
    cols = 4
    rows = (len(names) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * (tw + 20) + 20, rows * (th + 110) + 20), (245, 242, 236))
    d = ImageDraw.Draw(sheet)
    lines = []
    for n, name in enumerate(names):
        full = Image.open(next(os.path.join(ROOT, cfg["images_dir"], f) for f in
                               os.listdir(os.path.join(ROOT, cfg["images_dir"])) if f.startswith(name + ".")))
        keep = name in cfg["images"]
        im = load_source(cfg, name)[0] if keep else full.convert("RGB")
        x, y = 20 + (n % cols) * (tw + 20), 20 + (n // cols) * (th + 110)
        t = im.copy()
        t.thumbnail((tw, th))
        sheet.paste(t, (x + (tw - t.width) // 2, y))
        pal, w = palette(im)
        px = x
        for c, ww in zip(pal, w):
            d.rectangle((px, y + th + 8, px + int(tw * ww), y + th + 30), fill=c)
            px += int(tw * ww)
        d.text((x, y + th + 36), name[:34], fill=(30, 30, 30), font=font(15))
        d.text((x, y + th + 58), ("KEEP" if keep else "CUT") + f"  {full.width}x{full.height}",
               fill=(20, 120, 40) if keep else (180, 30, 30), font=font(15))
        lines.append(f"{'KEEP' if keep else 'CUT '} {name}: {full.width}x{full.height}, palette "
                     + " ".join("#%02x%02x%02x" % c for c in pal[:3])
                     + ("" if keep else f" -- {cfg['rejected'].get(name, '')}"))
    sheet.save(path, quality=90)
    return lines


def sequence_sheet(cfg, shots, plates, path, ver_name):
    seen, cells = [], []
    for s in shots:
        if s["image"] in seen and len(seen) == len(set(x["image"] for x in shots)):
            break
        seen.append(s["image"])
        cells.append(s)
    tw, th = 216, 384
    cols = min(len(cells), 4)
    rows = (len(cells) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * (tw + 16) + 16, rows * (th + 56) + 16), (20, 20, 20))
    d = ImageDraw.Draw(sheet)
    for n, s in enumerate(cells):
        x, y = 16 + (n % cols) * (tw + 16), 16 + (n // cols) * (th + 56)
        sheet.paste(Image.fromarray(plates[s["image"]].astype(np.uint8)).resize((tw, th)), (x, y))
        d.text((x, y + th + 6), f"{n + 1}. {s['image'][:24]}", fill=(235, 235, 235), font=font(13))
        d.text((x, y + th + 26), f"{s['start']:.2f}s  {'HOLD' if s['hold'] else ''}",
               fill=(240, 190, 120) if s["hold"] else (160, 160, 160), font=font(13))
    sheet.save(path, quality=90)


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(ROOT, "config.json"))
    ap.add_argument("--only")
    args = ap.parse_args()
    cfg = json.load(open(args.config))
    outdir = os.path.join(ROOT, "out", cfg["name"])
    os.makedirs(outdir, exist_ok=True)

    au = analyze_audio(cfg, None)
    print(f"tempo {au['tempo']:.1f} BPM" + ("  [PLACEHOLDER: no song file found at "
                                           f"{cfg['song']}]" if au["placeholder"] else
                                           f", {len(au['beats'])} beats, downbeat phase {au['phase']}"))
    for line in curation_sheet(cfg, os.path.join(outdir, "curation_sheet.jpg")):
        print(line)

    summary = []
    for name, ver in cfg["versions"].items():
        if args.only and name != args.only:
            continue
        mp4, shots, info, plates = render(cfg, name, ver, au, outdir)
        sequence_sheet(cfg, shots, plates, os.path.join(outdir, f"{name}_sequence.jpg"), name)
        rep, F = qa(cfg, mp4, shots, info, outdir, name)
        strip(F, [shots[i]["start_frame"] for i in (1, 2, 7) if i < len(shots)] + [len(F)],
              os.path.join(outdir, f"{name}_transition_strip.jpg"))
        s = (f"\n== {name} (preset {ver['preset']}) -> {os.path.relpath(mp4, ROOT)}\n"
             f"tempo {info['tempo']:.1f} BPM, base shot {info['subdivision_beats'] * ver.get('beat_scale', 1):g} beat(s) "
             f"({info['shot_seconds'] * ver.get('beat_scale', 1):.3f} s, holds 2x), {len(shots)} shots, {info['frames'] / cfg['fps']:.2f} s\n"
             f"window {info['window_start']:.2f}-{info['window_end']:.2f}s: {info['why']}\n" + "\n".join(rep))
        print(s)
        summary.append(s)
    with open(os.path.join(outdir, "summary.txt"), "w") as fh:
        fh.write("\n".join(summary) + "\n")


if __name__ == "__main__":
    main()
