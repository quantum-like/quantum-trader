#!/usr/bin/env python
"""fig_stills: 3 movies x 3 times, frames extracted from the mp4s with ffmpeg (report/figures/stills/*.png),
cropped to the box region and tiled with row/column labels."""
import sys, os, json
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, '/home/user/quantum-trader/fbdmd-hs-movies/report')
from figstyle import *  # noqa
apply_style()
plt.rcParams['axes.grid'] = False

STILLS = os.path.join(FIGDIR, 'stills')
MOVIES = ['hs_neutral', 'hs_charges_20p20m', 'hs_charges_20p20m_xray']
ROWLAB = ['neutral', 'charged CPK', 'charged x-ray']
FRAMES = [0, 360, 720]
TIMES = [0.0, 9.0, 18.0]
CROP = (320, 90, 960, 712)      # x0, y0, x1, y1 in the 1280x720 frame: box region, excludes overlay texts
cw, ch = CROP[2] - CROP[0], CROP[3] - CROP[1]

panel_w = 2.18
fig_h = 3 * panel_w * ch / cw * 1.03 + 0.35
fig, axs = plt.subplots(3, 3, figsize=(FULL_W, fig_h),
                        gridspec_kw=dict(left=0.055, right=0.995, top=0.955, bottom=0.005, wspace=0.03, hspace=0.04))
for i, (mv, rl) in enumerate(zip(MOVIES, ROWLAB)):
    for j, (k, t) in enumerate(zip(FRAMES, TIMES)):
        ax = axs[i, j]
        im = Image.open(os.path.join(STILLS, '%s_f%d.png' % (mv, k))).convert('RGB').crop(CROP)
        ax.imshow(np.asarray(im), interpolation='antialiased')
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(True); s.set_color('#CCCCD0'); s.set_linewidth(0.5)
        if i == 0:
            ax.text(0.5, 1.02, 't = %g τ' % t if j == 0 else '%g τ' % t, transform=ax.transAxes, ha='center', va='bottom', fontsize=9)
        if j == 0:
            ax.text(-0.035, 0.5, rl, transform=ax.transAxes, rotation=90, ha='right', va='center', fontsize=9)
            # label aligned with the TOP of its own row (panel_label's y = 1.04 would land in the row above with hspace 0.04)
            ax.text(-0.035, 1.0, '(%s)' % 'abc'[i], transform=ax.transAxes, fontsize=11, fontweight='bold', va='top', ha='right')
png = savefig(fig, 'fig_stills')
print('wrote', png, 'figure size', FULL_W, round(fig_h, 2))

piece = {
    'file': 'fig_stills', 'width': 'full',
    'caption': ('Stills taken from the three movies (frames 0, 360 and 720 of 721, i.e. t = 0, 9 and 18 τ of the NVE '
                'production run; the camera orbits from azimuth 25° to 65° over the movie): (a) the neutral '
                'control (1000 hard spheres, η = 0.35), (b) the charged run in the opaque CPK style (20 red + and 20 blue '
                '− ions, l_B = 5 σ), of which only the ions near the three camera-facing faces are visible, and (c) the '
                'same charged frames in the x-ray style with the neutrals 85 % transparent, which shows all 40 ions and '
                'their contact pairs inside the box.'),
    'numbers': {
        'frame_indices': FRAMES, 'times_tau': TIMES, 'frames_per_movie': 721, 'dt_frame_tau': 0.025,
        'crop_px_x0_y0_x1_y1': list(CROP), 'crop_size_px': [cw, ch], 'source_frame_px': [1280, 720],
        'figure_size_in': [FULL_W, round(fig_h, 2)],
        'camera_azimuth_deg_at_frames': [25.0, 45.0, 65.0], 'camera_elevation_deg': 22.0, 'camera_fov_deg': 30.0,
        'raw_frames_dir': 'report/figures/stills',
        'raw_frames': ['%s_f%d.png' % (mv, k) for mv in MOVIES for k in FRAMES],
    },
}
json.dump(piece, open(os.path.join(DATADIR, '_method_fig_stills.json'), 'w'), indent=1)
