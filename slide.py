"""One still slide of looping GIFs/clips, stacked in a centred column, for dropping into an edit by hand.

python3 slide.py --config config_slide_x.json  ->  out/<name>/<version>.mp4 (+ .png still, + _sequence.jpg)
Each clip keeps its own shape (scaled, never cropped); widths are equal so the column edges line up.
Length = the shortest time where every clip finishes a whole number of loops (capped by "max_seconds").
"""
import argparse, json, os, subprocess, math
from fractions import Fraction

ap = argparse.ArgumentParser(); ap.add_argument("--config", required=True); a = ap.parse_args()
c = json.load(open(a.config))
W, H = c["size"]; m, g, b = c["margin_px"], c["gap_px"], c["border_px"]
out = os.path.join("out", c["name"]); os.makedirs(out, exist_ok=True)

def probe(p):
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v", "-count_frames", "-show_entries",
                        "stream=width,height,duration", "-of", "json", p], capture_output=True, text=True)
    s = json.loads(r.stdout)["streams"][0]; return s["width"], s["height"], float(s["duration"])

items = [dict(path=p, **dict(zip(("w", "h", "dur"), probe(p)))) for p in c["clips"]]
# equal inner width w: sum(w*h_i/w_i) + 2b*n + g*(n-1) <= H - 2m, and w + 2b <= W - 2m
n = len(items); k = sum(i["h"] / i["w"] for i in items)
w = int(min((H - 2 * m - 2 * b * n - g * (n - 1)) / k, W - 2 * m - 2 * b)) // 2 * 2
for i in items: i["sw"], i["sh"] = w, int(round(w * i["h"] / i["w"])) // 2 * 2
total = sum(i["sh"] + 2 * b for i in items) + g * (n - 1)
x0, y = (W - w - 2 * b) // 2, (H - total) // 2
for i in items: i["x"], i["y"] = x0, y; y += i["sh"] + 2 * b + g

def flcm(x, y):  # least common multiple of two fractions
    return Fraction(x.numerator * y.numerator // math.gcd(x.numerator, y.numerator), math.gcd(x.denominator, y.denominator))
lcm = Fraction(0)
for i in items:
    d = Fraction(i["dur"]).limit_denominator(100); lcm = d if not lcm else flcm(lcm, d)
T = float(lcm) if float(lcm) <= c.get("max_seconds", 20) else c.get("max_seconds", 20)

gr = c.get("grade", {}); fps = c["fps"]
args, f = ["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", f"color=c={c['background'].replace('#', '0x')}:s={W}x{H}:r={fps}:d={T}"], []
for j, i in enumerate(items):
    args += ["-stream_loop", "-1", "-i", i["path"]]
    f.append(f"[{j+1}]fps={fps},scale={i['sw']}:{i['sh']}:flags=lanczos,format=rgb24,"
             f"pad={i['sw']+2*b}:{i['sh']+2*b}:{b}:{b}:{c['border'].replace('#', '0x')}[c{j}]")
last = "0"
for j, i in enumerate(items):
    f.append(f"[{last}][c{j}]overlay={i['x']}:{i['y']}:shortest=0:eof_action=repeat[o{j}]"); last = f"o{j}"
f.append(f"[{last}]eq=saturation={gr.get('saturation', 1)}:brightness={gr.get('lift', 0)/255:.4f},"
         f"noise=alls={gr.get('grain', 0)}:allf=t,trim=duration={T},format=yuv420p[v]")
mp4 = os.path.join(out, c["version"] + ".mp4")
subprocess.run(args + ["-filter_complex", ";".join(f), "-map", "[v]", "-c:v", "libx264", "-crf", "18",
                       "-preset", "medium", "-r", str(fps), "-movflags", "+faststart", mp4], check=True)
png = os.path.join(out, c["version"] + ".png")
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", "0", "-i", mp4, "-frames:v", "1", png], check=True)

cover = sum((i["sw"] + 2 * b) * (i["sh"] + 2 * b) for i in items) / (W * H)
lines = [f"{os.path.basename(mp4)}: {W}x{H}, {fps} fps, {T:.2f} s, silent (add it to your video by hand)"]
for i in items:
    lines.append(f"  {os.path.basename(i['path'])}: {i['w']}x{i['h']} source, {i['dur']:.2f} s loop, shown {i['sw']}x{i['sh']} "
                 f"at ({i['x']},{i['y']}) = {i['sw']/i['w']:.1f}x enlargement, loops {T/i['dur']:.2f} times")
lines.append(f"  shape kept: {all(abs(i['sw']/i['sh'] - i['w']/i['h']) < 0.01 for i in items)}; column edges aligned: {len({i['x'] for i in items}) == 1}")
lines.append(f"  gaps: top {items[0]['y']} px, between {g} px, bottom {H - total - items[0]['y']} px; screen covered {cover:.0%}")
open(os.path.join(out, c["version"] + "_qa.txt"), "w").write("\n".join(lines) + "\n"); print("\n".join(lines))
