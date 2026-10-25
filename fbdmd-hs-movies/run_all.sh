#!/usr/bin/env bash
# Reproduce both movies end to end (CPU only, no CUDA):
#   bash run_all.sh                 # full quality (default)
#   QUALITY=draft bash run_all.sh   # quick look (smaller/faster render)
#   FORCE=1 bash run_all.sh         # redo simulation + frames even if outputs exist
# Knobs (env): N ETA BJERRUM SEED T_RUN DT_FRAME DT INTEGRATOR(cv-phys|cv|kdk-phys|naive) ORDER_TEST(1|0) SELFTEST(1|0)
#              WIDTH HEIGHT FPS QUALITY RENDERER(ovito|numpy) XRAY(1|0)
# Plain bash >= 3.2 (stock macOS /bin/bash): no bash-4 features and no empty-array expansion under `set -u`.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

N=${N:-1000}; ETA=${ETA:-0.35}; BJERRUM=${BJERRUM:-5.0}; SEED=${SEED:-1}; INTEGRATOR=${INTEGRATOR:-cv-phys}
T_EQ_N=${T_EQ_N:-5}; T_EQ_C=${T_EQ_C:-15}; T_RUN=${T_RUN:-18}; DT_FRAME=${DT_FRAME:-0.025}; DT=${DT:-0.005}
WIDTH=${WIDTH:-1280}; HEIGHT=${HEIGHT:-720}; FPS=${FPS:-30}; QUALITY=${QUALITY:-final}
ORBIT=${ORBIT:-40}; XRAY=${XRAY:-1}; FORCE=${FORCE:-0}; SELFTEST=${SELFTEST:-1}
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
# requirements.txt = numpy + pillow (required) + ovito (optional primary renderer). If it cannot be satisfied as a
# whole (usually: no ovito wheel for this python/platform) install numpy + pillow alone; the renderer check below
# then falls back to render/render_numpy.py.
python -m pip install -q -r requirements.txt \
  || { echo "WARN: pip install -r requirements.txt failed (probably ovito); installing numpy pillow only"
       python -m pip install -q numpy pillow; }
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
if [ "$SELFTEST" = 1 ]; then
  log "sim/hsdmd --selftest (50 unit checks, ~7 s; SELFTEST=0 to skip)"
  # shows the summary line and any FAIL line; a non-zero exit of hsdmd aborts the run (pipefail)
  sim/hsdmd --selftest | grep -E 'FAIL|^selftest:'
fi

# ---------------------------------------------------------------- 2. simulate
mkdir -p traj movies frames notes
SIM_COMMON=(--N "$N" --eta "$ETA" --seed "$SEED" --t-equil-neutral "$T_EQ_N" --t-run "$T_RUN" --dt-frame "$DT_FRAME" --outdir traj)
if [ "$FORCE" = 1 ] || [ ! -f traj/hs_neutral.npz ]; then
  log "simulating hs_neutral (N=$N, eta=$ETA, EDMD)"
  python sim/run_sim.py --name hs_neutral "${SIM_COMMON[@]}"
else log "traj/hs_neutral.npz exists (FORCE=1 to redo)"; fi
if [ "$FORCE" = 1 ] || [ ! -f traj/hs_charges_20p20m.npz ]; then
  log "simulating hs_charges_20p20m (20+/20-, l_B=$BJERRUM sigma, integrator=$INTEGRATOR, dt=$DT)"
  python sim/run_sim.py --name hs_charges_20p20m "${SIM_COMMON[@]}" --nplus 20 --nminus 20 \
         --bjerrum "$BJERRUM" --dt "$DT" --t-equil-charged "$T_EQ_C" --coulomb ewald --integrator "$INTEGRATOR"
else log "traj/hs_charges_20p20m.npz exists (FORCE=1 to redo)"; fi

# ---------------------------------------------------------------- 3. sanity analysis (overlaps, energy, pairing)
log "analysis"
python sim/analyze.py traj/hs_neutral              | tee notes/analysis_hs_neutral.txt
python sim/analyze.py traj/hs_charges_20p20m       | tee notes/analysis_hs_charges_20p20m.txt

# ---------------------------------------------------------------- 3b. optional energy-order test (integrator comparison)
if [ "${ORDER_TEST:-0}" = 1 ]; then
  log "order test: max|E(t)-E(0)|/N vs dt for cv-phys / cv / kdk-phys / naive (equilibrated start, ~1 min)"
  python sim/order_test.py --name _test/order --N "$N" --eta "$ETA" --nplus 20 --nminus 20 --bjerrum "$BJERRUM" \
         --seed "$SEED" --t-equil-charged "$T_EQ_C" | tee notes/order_test.txt
  cp -f traj/_test/order_order.csv notes/order_test.csv 2>/dev/null || true
fi

# ---------------------------------------------------------------- 4. render
RENDER="render/render_${RENDERER}.py"
render(){ # name style label
  local name=$1 style=$2 label=$3 t0 t1
  local -a args
  # One argument array that is never empty: expanding an EMPTY array as "${a[@]}" under `set -u` is an
  # "unbound variable" error in bash < 4.4 (stock macOS /bin/bash is 3.2.57), so --force is appended here.
  args=(--traj "traj/${name%_xray}.npz" --name "$name" --out "movies/$name.mp4"
        --frames-dir "frames/$name" --width "$WIDTH" --height "$HEIGHT" --fps "$FPS"
        --style "$style" --quality "$QUALITY" --orbit-deg "$ORBIT" --label "$label")
  if [ "$FORCE" = 1 ]; then args+=(--force); fi
  log "rendering $name ($style) with $RENDER"
  t0=$(date +%s)
  python "$RENDER" "${args[@]}"
  t1=$(date +%s)
  echo "$name: $((t1-t0)) s wall for render+encode ($RENDERER, $QUALITY, ${WIDTH}x${HEIGHT})" >> notes/render_timing.txt
}
: > notes/render_timing.txt
render hs_neutral          cpk  "N=$N hard spheres, eta=$ETA, kT=1 (neutral control) - EDMD"
render hs_charges_20p20m   cpk  "N=$N hard spheres, eta=$ETA, 20+ / 20- (l_B=${BJERRUM} sigma) - collision Verlet"
if [ "$XRAY" = 1 ]; then
render hs_charges_20p20m_xray xray "N=$N hard spheres, eta=$ETA, 20+ / 20- (l_B=${BJERRUM} sigma) - collision Verlet; neutrals ghosted"
fi

# ---------------------------------------------------------------- 5. notes + summary
python make_notes.py
log "done"
for f in movies/*.mp4; do echo "$ROOT/$f"; done
