#!/usr/bin/env python
"""Assemble report/data/method_manifest.json from the per-figure pieces written by report/method_fig_*.py."""
import sys, os, json, datetime

sys.path.insert(0, '/home/user/quantum-trader/fbdmd-hs-movies/report')
from figstyle import DATADIR, FIGDIR

ORDER = ['fig_pipeline', 'fig_cv_scheme', 'fig_order', 'fig_stills', 'fig_pair_closeup', 'fig_render_cost']
NOTES = [
    'fig_stills is 7.2 x 6.9 in instead of the requested ~7.2 x 4.3 in: three rows of near-square box crops '
    '(640 x 622 px each) cannot fit a 4.3 in tall full-width figure without shrinking the spheres below legibility; '
    'the per-figure height convention (2.4-4.5 in) is therefore exceeded for this one figure.',
    'The DSF order-test data come from traj/_test/cv_confirmer_order/dsf_order.csv (identical copies exist in '
    'traj/_test/cv_impl and traj/_test/cv_fixer_order); only the cv-phys and cv DSF curves are drawn in fig_order(b) '
    'to keep the panel readable, the kdk-phys and naive DSF slopes are in the numbers.',
    'The integrator palette of figstyle (C_INTEG) is kept as required; because the gray "naive" entry and the '
    'orange/green pair are weak under colour-vision deficiency, every integrator additionally has its own marker '
    'shape and the slopes are printed in the legend.',
    'fig_pair_closeup: the three frames (651, 390, 369) were re-rendered with render/render_ovito.py at 3840 x 2160 '
    '(3x the movie, x-ray style, final quality, no overlay) into traj/_test/report/closeup_frames/ (scratch), so the '
    '900 x 792 px crops are shown at ~390 dpi; the camera pose is that of the movie frame. Panel (c) uses the linear '
    '5-ion chain of frame 369 (42 sigma deep, 29 neutrals in front) instead of frame 512 (48 sigma, 128 neutrals in '
    'front); all six-ion clusters are branched and 48 sigma deep. Particle ids in its numbers are given both as 0-based '
    'npz indices (member_particle_indices) and 1-based extended-XYZ ids (member_xyz_ids).',
    'fig_order caption: the ~1e-6 kT/N Ewald truncation floor limits only cv-phys at h = 0.0025 tau (local slope 1.0); '
    'cv and naive are flat there because of their own first-order collision error (65x and 229x above the floor); '
    'slopes in the captions are formatted from the unrounded log-log fits (slope_fit in numbers is rounded to 3 decimals).',
    'Every in-figure text of the method figures is >= 8 pt.',
]

figs = []
for name in ORDER:
    p = os.path.join(DATADIR, '_method_%s.json' % name)
    piece = json.load(open(p))
    for ext in ('png', 'svg'):
        f = os.path.join(FIGDIR, name + '.' + ext)
        assert os.path.exists(f), f
    piece['png'] = os.path.relpath(os.path.join(FIGDIR, name + '.png'), os.path.dirname(os.path.dirname(DATADIR)))
    piece['svg'] = os.path.relpath(os.path.join(FIGDIR, name + '.svg'), os.path.dirname(os.path.dirname(DATADIR)))
    figs.append(piece)

manifest = {
    'role': 'method / diagram / stills figures',
    'generated': datetime.datetime.now().strftime('%Y-%m-%d %H:%M'),
    'figure_dir': 'report/figures',
    'scripts': ['report/method_fig_%s.py' % n.replace('fig_', '') for n in ORDER] + ['report/method_manifest_build.py'],
    'raw_stills_dir': 'report/figures/stills',
    'figures': figs,
    'notes': NOTES,
}
out = os.path.join(DATADIR, 'method_manifest.json')
json.dump(manifest, open(out, 'w'), indent=1, ensure_ascii=False)
print(json.dumps(manifest, indent=1, ensure_ascii=False))
