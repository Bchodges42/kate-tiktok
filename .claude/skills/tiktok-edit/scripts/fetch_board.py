#!/usr/bin/env python3
"""Download every photo on a public Pinterest board, whole and at the largest size on offer.

    python3 fetch_board.py <board link or pin.it short link> --out <empty folder>

Writes NN.jpg (board order), board.json (one entry per pin: description, source site, size, and
flags for soft or video pins) and sheet-N.jpg contact sheets with the index printed on each photo,
so the photos can be looked at, named and tagged before any go into images/.

Nothing here crops or edits a photo. Network calls go through curl so the session proxy is used.
"""
import argparse
import html
import json
import os
import re
import subprocess
import sys

from PIL import Image, ImageDraw

UA = "Mozilla/5.0 (compatible; board-fetch)"


def curl(args, binary=False):
    r = subprocess.run(["curl", "-sS", "-L", "--max-time", "60", "-A", UA] + args, capture_output=True)
    if r.returncode != 0:
        raise RuntimeError(r.stderr.decode(errors="replace").strip())
    return r.stdout if binary else r.stdout.decode(errors="replace")


def resolve(link):
    """pin.it short link or full board URL -> (user, board slug)."""
    url = link.strip()
    if "pin.it/" in url:
        url = curl(["-o", "/dev/null", "-w", "%{url_effective}", url]).strip()
    m = re.search(r"pinterest\.[a-z.]+/([^/?#]+)/([^/?#]+)", url)
    if not m or m.group(1) in ("pin", "search", "ideas"):
        sys.exit(f"Not a board link: {url}\nOpen the board in Pinterest, tap Share, copy the link.")
    return m.group(1), m.group(2)


def pins_from_widget(user, board):
    raw = curl([f"https://api.pinterest.com/v3/pidgets/boards/{user}/{board}/pins/"])
    data = json.loads(raw).get("data") or {}
    out = []
    for p in data.get("pins") or []:
        imgs = p.get("images") or {}
        src = (imgs.get("564x") or next(iter(imgs.values()), {})).get("url")
        if not src:
            continue
        out.append(dict(id=p.get("id"), url=src, video=bool(p.get("is_video")),
                        description=html.unescape(p.get("description") or "").strip(),
                        source=p.get("domain") or "", link=html.unescape(p.get("link") or "")))
    return out, (data.get("board") or {}).get("pin_count")


def pins_from_rss(user, board):
    raw = curl([f"https://www.pinterest.com/{user}/{board}.rss"])
    out = []
    for item in re.findall(r"<item>(.*?)</item>", raw, flags=re.S):
        m = re.search(r'https://i\.pinimg\.com/[^"&<\s]+', html.unescape(item))
        if m:
            t = re.search(r"<title>(.*?)</title>", item, flags=re.S)
            out.append(dict(id=None, url=m.group(0), video=False, source="", link="",
                            description=html.unescape(re.sub(r"<!\[CDATA\[|\]\]>", "", t.group(1))).strip() if t else ""))
    return out, None


def download(url, dst):
    """Largest first: the original upload, then 736 px, then whatever the feed gave."""
    tried = []
    for size in ("originals", "736x", None):
        u = re.sub(r"pinimg\.com/[^/]+/", f"pinimg.com/{size}/", url, count=1) if size else url
        if u in tried:
            continue
        tried.append(u)
        try:
            blob = curl([u], binary=True)
            tmp = dst + ".part"
            open(tmp, "wb").write(blob)
            im = Image.open(tmp)
            im.load()
            im.convert("RGB").save(dst, quality=95)
            os.remove(tmp)
            return im.size
        except Exception:
            if os.path.exists(dst + ".part"):
                os.remove(dst + ".part")
    return None


def sheets(out, entries, cols=5, cell=360):
    per = cols * 4
    for s in range(0, len(entries), per):
        chunk = entries[s:s + per]
        rows = -(-len(chunk) // cols)
        sheet = Image.new("RGB", (cols * cell, rows * cell), "#222222")
        d = ImageDraw.Draw(sheet)
        for i, e in enumerate(chunk):
            im = Image.open(os.path.join(out, e["file"]))
            im.thumbnail((cell - 12, cell - 12))
            x = (i % cols) * cell + (cell - im.width) // 2
            y = (i // cols) * cell + (cell - im.height) // 2
            sheet.paste(im, (x, y))
            label = f'{e["index"]:02d} {e["width"]}x{e["height"]}' + (" SOFT" if e["soft"] else "")
            d.rectangle([x, y, x + 8 * len(label) + 8, y + 18], fill="#000000")
            d.text((x + 4, y + 3), label, fill="#FFFFFF")
        sheet.save(os.path.join(out, f"sheet-{s // per + 1}.jpg"), quality=88)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("link")
    ap.add_argument("--out", required=True, help="a new, empty folder (not images/)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if os.listdir(a.out):
        sys.exit(f"{a.out} is not empty; give a new folder so nothing is overwritten.")

    user, board = resolve(a.link)
    try:
        pins, count = pins_from_widget(user, board)
        via = "widget feed (newest 50 pins at most)"
    except Exception as e:
        pins, count, via = [], None, f"widget feed failed: {e}"
    if not pins:
        pins, count = pins_from_rss(user, board)
        via = "RSS feed (newest ~25 pins only)"
    if not pins:
        sys.exit(f"No pins found for {user}/{board}. The board may be secret or empty; ask the user to make it public or upload the photos.")

    entries = []
    for i, p in enumerate(pins):
        fn = f"{i:02d}.jpg"
        size = download(p["url"], os.path.join(a.out, fn))
        if not size:
            print(f"  {i:02d}: download failed, skipped")
            continue
        entries.append(dict(index=i, file=fn, width=size[0], height=size[1], soft=min(size) <= 500,
                            shape="tall" if size[1] > size[0] * 1.1 else "wide" if size[0] > size[1] * 1.1 else "square",
                            video_pin=p["video"], description=p["description"], source=p["source"], pin_link=p["link"]))
    json.dump(dict(user=user, board=board, url=f"https://www.pinterest.com/{user}/{board}/", via=via,
                   board_pin_count=count, fetched=len(entries), pins=entries),
              open(os.path.join(a.out, "board.json"), "w"), indent=1)
    sheets(a.out, entries)

    print(f"{user}/{board}: {len(entries)} photos via {via}")
    if count and count > len(entries):
        print(f"  The board has {count} pins; {count - len(entries)} are not in the feed. Tell the user which are missing if it matters.")
    soft = [e["index"] for e in entries if e["soft"]]
    if soft:
        print(f"  Soft (small) sources, flag or leave out: {soft}")
    vids = [e["index"] for e in entries if e["video_pin"]]
    if vids:
        print(f"  Video pins (only the cover frame was saved): {vids}")
    print(f"  Look at {a.out}/sheet-*.jpg, then copy the keepers into images/ with descriptive names.")


if __name__ == "__main__":
    main()
