#!/usr/bin/env python
"""fig_render_cost: seconds per 1280x720 frame on the 4-core box for the renderer presets
(render/README_render.md measured table) with the production-run wall times (notes/render_timing.txt)."""
import sys, os, json
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, '/home/user/quantum-trader/fbdmd-hs-movies/report')
from figstyle import *  # noqa
apply_style()

N_FRAMES = 721
# (label, s/frame incl. overlay+PNG from README_render.md "Measured" table, kind)
ROWS = [
    ('OVITO cpk, final\n(AO 16 samples)', 1.43, 'ovito'),
    ('OVITO x-ray, final', 1.27, 'ovito'),
    ('OVITO cpk, draft', 0.26, 'ovito'),
    ('numpy, 4 workers', 0.24, 'numpy'),
]
# production runs: wall seconds for render + encode of 721 frames (notes/render_timing.txt)
PROD = {'hs_neutral (cpk final)': (1028, 0), 'hs_charges_20p20m (cpk final)': (1031, 0),
        'hs_charges_20p20m_xray (x-ray final)': (896, 1)}
C_OVITO, C_NUMPY = '#E0A458', C_NEUTRAL

fig, ax = plt.subplots(figsize=(HALF_W, 2.9), gridspec_kw=dict(left=0.36, right=0.97, top=0.83, bottom=0.15))
y = np.arange(len(ROWS))[::-1]
vals = [r[1] for r in ROWS]
cols = [C_OVITO if r[2] == 'ovito' else C_NUMPY for r in ROWS]
ax.barh(y, vals, height=0.5, color=cols, edgecolor='none', zorder=2)
for yi, v in zip(y, vals):
    ax.text(v + 0.03, yi, '%.2f s  ≈ %.0f min' % (v, v * N_FRAMES / 60), va='center', ha='left', fontsize=8, color='#222222')
# production runs: ticks on the bars, annotated directly (a legend handle '|' reads as a fourth measurement)
for name, (wall, row) in PROD.items():
    ax.plot(wall / N_FRAMES, y[row], marker='|', ms=10, mew=1.4, color='#222222', ls='none', zorder=4)
ax.text(0.02, y[0] - 0.47, 'production movies: %.2f s, %.2f s (ticks)' % (1028 / N_FRAMES, 1031 / N_FRAMES), fontsize=8,
        color='#444444', va='center', ha='left')
ax.text(0.02, y[1] - 0.47, 'production movie: %.2f s (tick)' % (896 / N_FRAMES), fontsize=8, color='#444444', va='center', ha='left')
ax.set_ylim(-0.55, 3.55)
ax.set_yticks(y); ax.set_yticklabels([r[0] for r in ROWS], fontsize=8)
ax.set_xlim(0, 2.2)
ax.set_xlabel('seconds per 1280×720 frame')
ax.grid(axis='y', visible=False)
sec = ax.secondary_xaxis('top', functions=(lambda s: s * N_FRAMES / 60, lambda m: m * 60 / N_FRAMES))
sec.set_xlabel('minutes per 721-frame movie', fontsize=8)
sec.tick_params(labelsize=8)
png = savefig(fig, 'fig_render_cost')
print('wrote', png)

prod_s = {k: v[0] / N_FRAMES for k, v in PROD.items()}
piece = {
    'file': 'fig_render_cost', 'width': 'half',
    'caption': ('Rendering cost per 1280×720 frame on the 4-core Xeon box (no GPU), measured on the 1000-sphere test '
                'trajectory including overlay and PNG writing: OVITO/Tachyon cpk at final quality (16 ambient-occlusion '
                'samples, 16 Tachyon threads) takes 1.43 s per frame, i.e. ≈ 17 min for a 721-frame movie, the x-ray '
                'preset 1.27 s (≈ 15 min), the AO-free draft preset 0.26 s (≈ 3 min) and the numpy fallback renderer '
                'with 4 worker processes 0.24 s wall (≈ 3 min). The annotated vertical ticks mark the three production '
                'movies: 1028 s and 1031 s wall for render plus encode of the two cpk movies (1.43 s per frame, two coinciding '
                'ticks) and 896 s for the x-ray movie (1.24 s per frame).'),
    'numbers': {
        'seconds_per_frame_incl_overlay_png': {r[0].replace('\n', ' '): r[1] for r in ROWS},
        'minutes_per_721_frame_movie_estimate': {r[0].replace('\n', ' '): round(r[1] * N_FRAMES / 60, 1) for r in ROWS},
        'production_wall_s_render_plus_encode': {k: v[0] for k, v in PROD.items()},
        'production_s_per_frame_incl_encode': {k: round(v, 3) for k, v in prod_s.items()},
        'production_minutes': {k: round(v[0] / 60, 1) for k, v in PROD.items()},
        'render_only_s_per_frame_readme': {'ovito cpk final AO16 16 threads': 1.38, 'ovito cpk final AO16 4 threads': 2.02,
                                           'ovito cpk final AO6': 0.65, 'ovito cpk draft': 0.20, 'ovito xray final': 1.22,
                                           'ovito cpk final 1920x1080': 3.49, 'numpy cpk final 1 process': 0.56},
        'ovito_thread_count': 16, 'cpu': '4 x Intel Xeon 2.10 GHz, no GPU',
        'ffmpeg_encode_s_per_721_frames': 20, 'xray_ao_on_rejected_s_per_frame': 12.3,
    },
}
json.dump(piece, open(os.path.join(DATADIR, '_method_fig_render_cost.json'), 'w'), indent=1)
