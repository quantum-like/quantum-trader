# render/ — Materials-Studio-style CPK movies from FORMAT.md trajectories

Two renderers with an identical CLI:

| script | backend | 1280x720 s/frame (4-core x86, N = 1000) | use |
|---|---|---|---|
| `render_ovito.py` | OVITO 3.17 Python module, TachyonRenderer (ray tracing, AO, shadows), 16 Tachyon threads (see below) | final cpk **1.4 s** (AO 16 samples), draft cpk **0.26 s**, final xray **1.3 s** | primary |
| `render_numpy.py` | numpy + Pillow only (software rasteriser) | **0.6 s** single process, **0.24 s wall** with `--jobs 4` (cpk or xray) | fallback when `import ovito` fails (e.g. Mac without a working Vulkan/Qt offscreen stack) |

Both read `traj/<name>.npz` (preferred) or the extended-XYZ `traj/<name>.xyz`, write
`frames/<name>/frame_%05d.png`, encode with ffmpeg to H.264/yuv420p mp4 (`-crf 18 -preset slow
-movflags +faststart`), verify with ffprobe (printed as JSON) and, unless `--keep-frames`, delete the PNGs
after a verified encode. Frame rendering is resumable: PNGs that already exist are skipped unless `--force`.

Estimated wall time for a 721-frame movie (24 s at 30 fps) on this 4-core box: OVITO final cpk ~17 min,
OVITO final xray ~15 min, OVITO draft ~3 min, numpy `--jobs 4` ~3 min (+ ~20 s ffmpeg `-preset slow`).
The three production movies of `run_all.sh` (cpk neutral, cpk charged, xray charged) therefore take ~50 min
at 720p final quality; `--ao-samples 6` brings the two cpk movies to ~8.5 min each at the price of a visible
AO grain (see "Look").

## Commands

```bash
. .venv/bin/activate            # python 3.13, numpy, pillow, ovito 3.17
python render/make_test_traj.py  # -> traj/_test/synthetic.{npz,xyz,json}  (12 frames, 1000 spheres, 20+/20-)

# primary
python render/render_ovito.py --traj traj/hs_neutral.npz --name hs_neutral \
    --out movies/hs_neutral.mp4 --frames-dir frames/hs_neutral \
    --width 1280 --height 720 --fps 30 --style cpk --bg light --quality final \
    --orbit-deg 40 --elev-deg 22 --label "1000 hard spheres, no charges"

python render/render_ovito.py --traj traj/hs_charges_20p20m.npz --name hs_charges_20p20m \
    --out movies/hs_charges_20p20m.mp4 --frames-dir frames/hs_charges_20p20m --style cpk
python render/render_ovito.py --traj traj/hs_charges_20p20m.npz --name hs_charges_20p20m_xray \
    --out movies/hs_charges_20p20m_xray.mp4 --frames-dir frames/hs_charges_20p20m_xray --style xray

# fallback (same flags, plus --jobs)
python render/render_numpy.py --traj traj/hs_neutral.npz --name hs_neutral --jobs 4

# pattern for run scripts
python -c "import ovito" 2>/dev/null && R=render/render_ovito.py || R=render/render_numpy.py
python $R --traj traj/hs_neutral.npz --name hs_neutral
```

Full option list (`--help`):
`--traj --name --out --frames-dir --width 1280 --height 720 --fps 30 --style cpk|xray --bg light|dark
--quality draft|final --orbit-deg 40 --elev-deg 22 --azim0-deg (default 45-orbit/2) --fov-deg 30
--start 0 --end -1 --stride 1 --label "text" ('' = none; default built from metadata) --no-legend
--no-overlay --no-encode --keep-frames --force --crf 18 --preset slow`.
`render_ovito.py` adds `--ao-samples --aa-samples --ao-brightness --no-shadows --threads`; `render_numpy.py`
adds `--jobs`. Relative paths are resolved from the current directory (run from the project root); defaults
are `<root>/movies/<name>.mp4` and `<root>/frames/<name>`.

### Frame ranges, chunks and previews (`--start/--end/--stride`)

* The output frame (PNG) index and the camera azimuth are always those of the **full** movie:
  `--stride s` renders trajectory frames 0, s, 2s, ... and trajectory frame k becomes `frame_{k//s:05d}.png`
  with the azimuth of that position in the orbit, independent of `--start/--end`. A `--start` that is not a
  multiple of the stride starts at the next grid frame.
* The mp4 of a run holds exactly that run's frames (ffmpeg is called with `-start_number j0 -frames:v n`, so
  stale PNGs next to them are never picked up). With a partial range the script prints a NOTE that the movie
  covers frames j0..j1 of the N-frame movie — useful for a quick preview (`--end 29`).
* To render a movie in chunks (e.g. on two machines or over several sessions): run each chunk with
  `--keep-frames` into the same `--frames-dir`, then run once more without `--start/--end`; that last run
  skips the existing PNGs and encodes all of them. Chunks rendered this way are bit-identical to the frames
  of a single full run (tested for both renderers).

### Encode and verification

* `ffmpeg`/`ffprobe` presence is checked right after argument parsing (before any rendering); without
  ffmpeg the script exits immediately unless `--no-encode` is given.
* After encoding, ffprobe must report h264 / yuv420p / exactly the expected number of frames. If not, the mp4 is
  renamed to `<out>.unverified.mp4`, the PNGs are kept and the script exits non-zero (so `run_all.sh`,
  which uses `set -e`, stops instead of continuing with a wrong movie).

## Files

* `render_ovito.py` — OVITO/Tachyon renderer. ONE pipeline: `StaticSource` with the frame-0
  `DataCollection` (cell, particle types/colours/radii, optional `Transparency`) plus a Python modifier
  that overwrites the `Position` array with `traj.pos[frame]` for the frame passed to
  `render_image(frame=k)`; nothing is re-imported per frame. The rendered `QImage` is converted to numpy,
  captioned with Pillow and written as PNG. Sets `OVITO_THREAD_COUNT` (default 4 x cores, `--threads N`)
  before `import ovito`.
* `render_numpy.py` — software renderer: per-sphere screen-bounding-box rasterisation with analytic
  ray–sphere intersection, z-buffer (ray distance from the camera), back-to-front order, analytic silhouette
  antialiasing, Blinn–Phong (key + fill + ambient + specular), ambient-occlusion cue = neighbour count within
  1.5 σ (minimum image) darkening the ambient term, box edges as splatted 3-D lines tested against the same
  ray-distance z-buffer, xray via alpha compositing of the 85 % transparent neutrals over opaque ions.
  `--jobs N` splits frames round-robin over `spawn` worker processes (portable to macOS).
* `common.py` — shared CLI, colours, `Camera` (orbit + fit), Pillow overlay, resumable frame loop
  (absolute frame numbering), tool check, ffmpeg/ffprobe helpers.
* `trajio.py` — `.npz` / extended-XYZ readers and writers per FORMAT.md.
* `make_test_traj.py` — synthetic 12-frame test trajectory (jittered simple-cubic lattice, L = 11.444,
  40 ions, ballistic drift with wrapping; min centre distance in frame 0 = 1.049 σ).
* `_ovito_smoketest.py` — original headless smoke test (kept for reference).

## Look (Materials Studio CPK)

* Spheres: radius 0.5 σ, opaque. Type 1 `HS` gray (0.78, 0.78, 0.80), type 2 `POS` red
  (0.85, 0.15, 0.15), type 3 `NEG` blue (0.15, 0.35, 0.90).
* Background light (0.97, 0.97, 0.98) by default, `--bg dark` = (0.07, 0.08, 0.10).
* Periodic cell drawn as thin lines, width 0.035·(L/11.4) σ (~1.7 px at 720p), colour (0.35, 0.35, 0.40) on
  light / (0.75, 0.75, 0.80) on dark. Lines are depth-tested geometry in both renderers, so spheres
  straddling a face hide the edge where they protrude (as in OVITO/MS). In the numpy renderer the test
  compares the line point's distance from the camera with the z-buffer, which stores the ray-sphere
  intersection distance (an earlier version compared view-axis depth against it and drew edge stubs through
  spheres near the box periphery; fixed).
* Camera: perspective, vertical FOV 30°, elevation 22°, azimuth sweeping `--orbit-deg` (default 40°)
  centred on the 45° three-quarter view (25° → 65°), z up. The distance to the box centre is fitted
  ONCE analytically so that the box expanded by one sphere radius fits with a 5 % margin at every
  azimuth of the orbit (`Camera.fit_distance`); the camera never re-zooms or pops. The script prints the
  worst-case projected-corner margin (18 px at 720p). OVITO's own `zoom_all` on frame 0 is printed for
  comparison only (it would be ~5 % closer and would clip at the orbit ends).
* Tachyon `final` (cpk): ambient occlusion **16 samples** (brightness 0.8), shadows, 4× antialiasing, direct
  light 0.9. AO 6 samples (the first version) leaves a streaky, screen-fixed grain on every sphere that the
  spheres slide under as they move; measured against a 48-sample reference on sphere pixels of the synthetic
  frame 0 (1280x720): AO 6 → noise std 10.4 levels, 42.8 % of sphere pixels off by > 8 levels;
  AO 16 → 4.6 levels, 10.4 %; AO 24 → 4.4 / 7.9 %. Cost at 16 threads: AO 6 0.70 s/frame, AO 16 1.43 s,
  AO 48 4.0 s. Override with `--ao-samples N`; `draft` (cpk): AO off, shadows off, 2× AA (flat, grain-free,
  also a legitimate "MS CPK" look).
  `xray` presets: AO **off**, shadows **on**, AA 4× (final) / 2× (draft), `max_ray_recursion` 50.
  Reasons (measured): AO with 960 transparent spheres costs 11–12 s/frame; `max_ray_recursion` ≤ 16
  fragments the interior ions (rays die inside the transparent stack); AO off together with shadows
  off makes Tachyon drop the transparent spheres entirely. `--no-shadows` is therefore ignored for xray.
* numpy renderer lighting (camera space): key light from upper-left-front (0.72), fill from lower-right
  (0.22), ambient 0.45 × AO factor (1 − 0.55·min(n₁.₅σ, 12)/12), specular 0.30 with shininess 40.
* Overlay (Pillow, DejaVu Sans / Arial / Helvetica, else PIL default): `--label` top-left
  (default e.g. "N = 1000 hard spheres   η = 0.35   20 + / 20 − ions   l_B/σ = 5", from the npz metadata),
  "t = 12.35 τ" bottom-left (times from the trajectory), small colour legend bottom-right; dark-gray text on
  the light background, light text on dark.

## Tachyon threads (`OVITO_THREAD_COUNT`, `--threads`)

Tachyon's default thread count (= number of cores) keeps only ~2 of the 4 cores busy here: about 0.5 s of
each 720p frame is single-threaded per-pixel work that the worker threads wait on. Oversubscribing helps:
`render_ovito.py` sets `OVITO_THREAD_COUNT = 4 x os.cpu_count()` before `import ovito` unless the variable is
already set or `--threads N` is given, and prints the value on the `tachyon:` line. Measured (720p, N = 1000,
12 frames, incl. overlay + PNG): cpk final AO 16: 2.07 s/frame with 4 threads → 1.43 s with 16; cpk final
AO 6: 1.25 → 0.70; xray final: 1.75 → 1.27; cpk draft: 0.66 → 0.26. 32 threads gives no further gain, 2 is
slower. Frames rendered with 4 and 16 threads are bit-identical. Two concurrent renders (0.80 s/frame
effective with default threads) are slower than one 16-thread render, so still run one at a time.

## Measured (this box: Linux x86_64, 4 CPUs, no GPU; synthetic 1000-sphere test, 12 frames at 1280x720)

| run | s/frame render | s/frame incl. overlay + PNG | 721 frames |
|---|---|---|---|
| ovito cpk final (AO 16, 16 threads) | 1.38 | 1.43 | ~17 min |
| ovito cpk final (AO 16, 4 threads = old default) | 2.02 | 2.07 | ~25 min |
| ovito cpk final with `--ao-samples 6` (16 threads) | 0.65 | 0.70 | ~8.5 min |
| ovito cpk draft (16 threads) | 0.20 | 0.26 | ~3 min |
| ovito cpk final 1920x1080 (AO 16, 16 threads) | 3.49 | 3.60 | ~43 min |
| ovito xray final (16 threads) | 1.22 | 1.27 | ~15 min |
| ovito xray with AO on (rejected) | 12.3 | — | — |
| numpy cpk final, 1 process | 0.55–0.57 | 0.59–0.62 | ~7.5 min |
| numpy cpk draft, 1 process | 0.54 | 0.58 | ~7 min |
| numpy xray final, 1 process | 0.55 | 0.60 | ~7.5 min |
| numpy cpk final, `--jobs 4` | — | 0.24 wall | ~3 min |
| ffmpeg libx264 crf 18 preset slow, 12 frames | — | ~1 s | ~20 s |

Real simulator output (`traj/_test/charged_small.npz`, N = 216, 161 frames) also rendered fine with both
scripts (ovito final 1.07 s/frame at AO 6 / default threads, numpy 0.24 s/frame).

Checks done on the PNGs: red and blue pixels present in every variant (e.g. 3.5k red / 1.7k blue pixels in
the cpk frame, more in xray), top-right corner pixel = (247, 247, 250) for light and (18, 20, 26) for dark,
170k–190k pixels change between consecutive frames (orbit + motion), ffprobe of every test mp4:
h264, yuv420p, 1280x720, 30 fps, exact frame count. The xyz fallback (`--traj traj/_test/synthetic.xyz`)
gives the same frames as the npz (numpy renderer: max pixel difference 4/255 from the 6-decimal xyz
positions; Tachyon adds its own stochastic antialiasing noise at sphere edges). `--help` works without
importing ovito. Chunk tests: `--start 0 --end 5` then `--start 6 --end 11` into one frames dir give 6-frame
mp4s that differ, PNGs numbered 0..11 bit-identical to a full run, and a final run without range encodes all
12; with 12 PNGs present `--end 7` yields an 8-frame mp4; a simulated ffprobe mismatch exits 1 and leaves
`<out>.unverified.mp4`; with ffmpeg off PATH the script exits before rendering.

## Known limitations

* OVITO render cost scales with resolution (1080p ≈ 2.5× the 720p time) and with the number of transparent
  spheres. Run one OVITO render at a time (see Tachyon threads).
* AO 16 samples is still Monte-Carlo: a faint residual grain (~4.6 levels std) remains; `--ao-samples 24`
  (~2.4 s/frame at default threads) or the grain-free `draft` preset are the alternatives.
* The numpy renderer has no shadows and only an approximate AO; it is intended to "read as the same movie",
  not to be pixel-identical. Its antialiasing is analytic on sphere silhouettes only (box lines are splatted).
* Frames are deleted after a verified encode unless `--keep-frames`; pass `--keep-frames` if you want to
  re-encode with other ffmpeg settings or to stitch chunks.
* Overlay fonts: DejaVu Sans on Linux; on macOS Arial/Helvetica is used (the "τ"/"η"/"σ" glyphs exist in both).
