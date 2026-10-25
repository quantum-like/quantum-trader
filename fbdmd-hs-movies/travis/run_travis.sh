#!/usr/bin/env bash
# Start TRAVIS on an exported trajectory (interactive; records input.txt for replay). Usage:
#   bash travis/run_travis.sh traj/hs_charges_20p20m [input.txt]
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT"
BASE=${1:?trajectory base name, e.g. traj/hs_charges_20p20m}; INPUT=${2:-}
command -v travis >/dev/null || { echo "travis not on PATH (install from http://www.travis-analyzer.de, 'make' then copy exe/travis to PATH)"; exit 1; }
[ -f "$BASE"_elem.xyz ] || { . .venv/bin/activate; python tools/export_vmd_travis.py "$BASE".npz; }
L=$(python3 -c "import json;print('%.2f'%(json.load(open('$BASE.json'))['L']*400))")
mkdir -p travis/out_$(basename "$BASE"); cd travis/out_$(basename "$BASE")
echo "cubic cell edge for TRAVIS: $L pm   (1 sigma = 400 pm; frame spacing = 25 fs => 1 ps = 1 tau)"
if [ -n "$INPUT" ]; then [ -f "$INPUT" ] || INPUT="$ROOT/$INPUT"; travis -p "$ROOT/${BASE}_elem.xyz" -i "$INPUT"; else travis -p "$ROOT/${BASE}_elem.xyz"; fi
echo "note: TRAVIS wrote/overwrote input.txt in $(pwd); rename it (rdf.in, aggr.in, ...) to keep it"
