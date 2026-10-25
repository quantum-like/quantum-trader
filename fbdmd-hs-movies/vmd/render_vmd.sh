#!/usr/bin/env bash
# Render a movie with VMD + its bundled Tachyon instead of OVITO.   Usage:
#   bash vmd/render_vmd.sh traj/hs_charges_20p20m.npz hs_charges_20p20m_vmd [cpk|xray] [W H] [orbit_deg]
# Needs: VMD on PATH (or VMD=/Applications/VMD.app/Contents/MacOS/startup.command), ffmpeg, the project venv for the export.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT"
NPZ=${1:?npz}; NAME=${2:?output name}; STYLE=${3:-cpk}; W=${4:-1280}; H=${5:-720}; ORBIT=${6:-40}
VMD=${VMD:-vmd}
command -v "$VMD" >/dev/null || { echo "VMD not found: set VMD=/Applications/VMD*.app/Contents/MacOS/startup.command (macOS) or put vmd on PATH"; exit 1; }
LMP="${NPZ%.npz}.lammpstrj"
[ -f "$LMP" ] || { . .venv/bin/activate; python tools/export_vmd_travis.py "$NPZ"; }
mkdir -p "frames/$NAME" movies
"$VMD" -dispdev text -e vmd/render_cpk.tcl -args "$LMP" "frames/$NAME" "$STYLE" "$W" "$H" "$ORBIT"
ffmpeg -y -loglevel error -framerate 30 -i "frames/$NAME/frame_%05d.tga" -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" \
       -c:v libx264 -pix_fmt yuv420p -crf 18 -preset slow -movflags +faststart "movies/$NAME.mp4"
echo "movie: $ROOT/movies/$NAME.mp4"
