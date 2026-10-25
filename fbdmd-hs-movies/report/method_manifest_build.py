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
