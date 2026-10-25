#!/usr/bin/env bash
# Reproduce both movies end to end (CPU only, no CUDA):
#   bash run_all.sh                 # full quality (default)
#   QUALITY=draft bash run_all.sh   # quick look (smaller/faster render)
#   FORCE=1 bash run_all.sh         # redo simulation + frames even if outputs exist
# Knobs (env): N ETA BJERRUM SEED T_RUN DT_FRAME DT WIDTH HEIGHT FPS QUALITY RENDERER(ovito|numpy) XRAY(1|0)
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

N=${N:-1000}; ETA=${ETA:-0.35}; BJERRUM=${BJERRUM:-5.0}; SEED=${SEED:-1}
T_EQ_N=${T_EQ_N:-5}; T_EQ_C=${T_EQ_C:-15}; T_RUN=${T_RUN:-18}; DT_FRAME=${DT_FRAME:-0.025}; DT=${DT:-0.005}
WIDTH=${WIDTH:-1280}; HEIGHT=${HEIGHT:-720}; FPS=${FPS:-30}; QUALITY=${QUALITY:-final}
ORBIT=${ORBIT:-40}; XRAY=${XRAY:-1}; FORCE=${FORCE:-0}
PY=${PYTHON:-python3}

log(){ printf '\n\033[1;34m== %s\033[0m\n' "$*"; }

# ---------------------------------------------------------------- 0. environment
if [ ! -x .venv/bin/python ]; then
  log "creating virtualenv .venv"
  "$PY" -m venv .venv
fi
# shellcheck disable=SC1091
. .venv/bin/activate
python -m pip install -q --upgrade pip
python -m pip install -q numpy pillow
# OVITO is optional (primary renderer). If it cannot be installed/imported we fall back to the numpy renderer.
python -m pip install -q "ovito>=3.10" || echo "WARN: pip install ovito failed; will use numpy renderer"
command -v ffmpeg >/dev/null || { echo "ERROR: ffmpeg not found on PATH (brew install ffmpeg)"; exit 1; }

if [ -z "${RENDERER:-}" ]; then
  if python - <<'PYEOF' 2>/dev/null
import ovito, sys
from ovito.vis import Viewport, TachyonRenderer
sys.exit(0)
PYEOF
  then RENDERER=ovito; else RENDERER=numpy; fi
fi
log "renderer: $RENDERER   quality: $QUALITY   size: ${WIDTH}x${HEIGHT}@${FPS}"

# ---------------------------------------------------------------- 1. build simulator
log "building sim/hsdmd"
make -s -C sim

# ---------------------------------------------------------------- 2. simulate
mkdir -p traj movies frames notes
SIM_COMMON=(--N "$N" --eta "$ETA" --seed "$SEED" --t-equil-neutral "$T_EQ_N" --t-run "$T_RUN" --dt-frame "$DT_FRAME" --outdir traj)
if [ "$FORCE" = 1 ] || [ ! -f traj/hs_neutral.npz ]; then
  log "simulating hs_neutral (N=$N, eta=$ETA, EDMD)"
  python sim/run_sim.py --name hs_neutral "${SIM_COMMON[@]}"
else log "traj/hs_neutral.npz exists (FORCE=1 to redo)"; fi
if [ "$FORCE" = 1 ] || [ ! -f traj/hs_charges_20p20m.npz ]; then
  log "simulating hs_charges_20p20m (20+/20-, l_B=$BJERRUM sigma, impulsive Verlet dt=$DT)"
  python sim/run_sim.py --name hs_charges_20p20m "${SIM_COMMON[@]}" --nplus 20 --nminus 20 \
         --bjerrum "$BJERRUM" --dt "$DT" --t-equil-charged "$T_EQ_C" --coulomb ewald
else log "traj/hs_charges_20p20m.npz exists (FORCE=1 to redo)"; fi

# ---------------------------------------------------------------- 3. sanity analysis (overlaps, energy, pairing)
log "analysis"
python sim/analyze.py traj/hs_neutral              | tee notes/analysis_hs_neutral.txt
python sim/analyze.py traj/hs_charges_20p20m       | tee notes/analysis_hs_charges_20p20m.txt

# ---------------------------------------------------------------- 4. render
RENDER="render/render_${RENDERER}.py"
FORCEFLAG=(); [ "$FORCE" = 1 ] && FORCEFLAG=(--force)
render(){ # name style label
  local name=$1 style=$2 label=$3 t0 t1
  log "rendering $name ($style) with $RENDER"
  t0=$(date +%s)
  python "$RENDER" --traj "traj/${name%_xray}.npz" --name "$name" --out "movies/$name.mp4" \
      --frames-dir "frames/$name" --width "$WIDTH" --height "$HEIGHT" --fps "$FPS" \
      --style "$style" --quality "$QUALITY" --orbit-deg "$ORBIT" --label "$label" "${FORCEFLAG[@]}"
  t1=$(date +%s)
  echo "$name: $((t1-t0)) s wall for render+encode ($RENDERER, $QUALITY, ${WIDTH}x${HEIGHT})" >> notes/render_timing.txt
}
: > notes/render_timing.txt
render hs_neutral          cpk  "N=$N hard spheres, eta=$ETA, kT=1 (neutral control) - EDMD"
render hs_charges_20p20m   cpk  "N=$N hard spheres, eta=$ETA, 20+ / 20- (l_B=${BJERRUM} sigma) - impulsive Verlet"
if [ "$XRAY" = 1 ]; then
render hs_charges_20p20m_xray xray "same run, neutrals ghosted to show the ions"
fi

# ---------------------------------------------------------------- 5. notes + summary
python make_notes.py
log "done"
for f in movies/*.mp4; do echo "$ROOT/$f"; done
