#!/usr/bin/env bash
# Render a movie with VMD + its bundled Tachyon instead of OVITO.   Usage:
#   bash vmd/render_vmd.sh traj/hs_charges_20p20m.npz hs_charges_20p20m_vmd [cpk|xray] [W H] [orbit_deg] [aa]
# Needs: ffmpeg and VMD (vmd on PATH, or VMD=/Applications/VMD*.app/Contents/MacOS/startup.command; auto-detected on macOS).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT"
NPZ=${1:?npz}; NAME=${2:?output name}; STYLE=${3:-cpk}; W=${4:-1280}; H=${5:-720}; ORBIT=${6:-40}; AA=${7:-8}
if [ -z "${VMD:-}" ]; then
  if command -v vmd >/dev/null; then VMD=vmd
  else VMD=$(ls -d /Applications/VMD*.app/Contents/MacOS/startup.command 2>/dev/null | head -1 || true); fi
fi
[ -n "${VMD:-}" ] && [ -e "$VMD" -o -n "$(command -v "$VMD" 2>/dev/null)" ] || { echo "VMD not found: put vmd on PATH or set VMD=/Applications/VMD.app/Contents/MacOS/startup.command"; exit 1; }
command -v ffmpeg >/dev/null || { echo "ffmpeg not found (brew install ffmpeg)"; exit 1; }
LMP="${NPZ%.npz}.lammpstrj"
if [ ! -f "$LMP" ]; then . .venv/bin/activate; python tools/export_vmd_travis.py "$NPZ"; fi
OUT="$ROOT/frames/$NAME"; mkdir -p "$OUT" movies
rm -f "$OUT"/frame_*.tga "$OUT"/frame_*.dat
"$VMD" -dispdev text -e "$ROOT/vmd/render_cpk.tcl" -args "$ROOT/$LMP" "$OUT" "$STYLE" "$W" "$H" "$ORBIT" "$AA"
n=$(ls "$OUT"/frame_*.tga 2>/dev/null | wc -l | tr -d ' '); [ "$n" -gt 0 ] || { echo "no frames rendered in $OUT"; exit 1; }
ffmpeg -y -loglevel error -framerate 30 -i "$OUT/frame_%05d.tga" -frames:v "$n" -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" \
       -c:v libx264 -pix_fmt yuv420p -crf 18 -preset slow -movflags +faststart "movies/$NAME.mp4"
echo "movie: $ROOT/movies/$NAME.mp4  ($n frames)"
