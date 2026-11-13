#!/usr/bin/env bash
# Make the light 720p preview and its poster frame for the shared viewing page.
#   make_preview.sh out/<name>/<version>.mp4 <output folder>
set -euo pipefail
src="$1"; dst="$2"; base="$(basename "${src%.mp4}")"
mkdir -p "$dst"
ffmpeg -y -loglevel error -i "$src" -vf scale=720:1280 -c:v libx264 -crf 23 -preset veryfast \
  -pix_fmt yuv420p -c:a aac -b:a 128k -movflags +faststart "$dst/$base-720.mp4"
ffmpeg -y -loglevel error -i "$src" -frames:v 1 -vf scale=540:960 -q:v 3 "$dst/$base-poster.jpg"
ls -la "$dst/$base-720.mp4" "$dst/$base-poster.jpg"
