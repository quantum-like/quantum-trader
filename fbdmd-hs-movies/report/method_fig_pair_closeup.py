#!/usr/bin/env python
"""fig_pair_closeup: crops from movies/hs_charges_20p20m_xray.mp4 showing a +/- contact pair and ion chains.
Frames were chosen with a cluster scan of traj/hs_charges_20p20m.npz (traj/_test/report/cluster_scan.json,
bond criterion r(+-) < 1.2 sigma as in sim/analyze.py); the ions are located in the frame with the renderer's
Camera (render/common.py, read-only import) and the mp4 frames are extracted by index with ffmpeg."""
import sys, os, json, subprocess
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from PIL import Image

sys.path.insert(0, '/home/user/quantum-trader/fbdmd-hs-movies/report')
from figstyle import *  # noqa
apply_style()
plt.rcParams['axes.grid'] = False
ROOT = '/home/user/quantum-trader/fbdmd-hs-movies'
sys.path.insert(0, ROOT + '/render')
from common import Camera  # noqa  (read-only use of the renderer's camera model)

SCAN = ROOT + '/traj/_test/report/cluster_scan.json'
MOVIE = ROOT + '/movies/hs_charges_20p20m_xray.mp4'
STILLS = os.path.join(FIGDIR, 'stills')
BOND = 1.2
# (frame index K, cluster size to show, short label)
SEL = [(433, 2, '+/− contact pair'), (390, 4, '4-ion chain'), (512, 5, '5-ion chain')]
HALF_W_PX, HALF_H_PX = 150, 132          # crop half-sizes (px of the 1280x720 frame)

rows = json.load(open(SCAN))
d = np.load(ROOT + '/traj/hs_charges_20p20m.npz', allow_pickle=False)
L = float(d['L']); times = d['times']; F = len(times)
cam = Camera(L, 1280, 720, nout=F)
TAN = np.tan(np.radians(cam.fov and 15.0))      # vertical half-FOV (30 deg / 2)


def extract(k):
    out = os.path.join(STILLS, 'hs_charges_20p20m_xray_f%d.png' % k)
    if not os.path.exists(out):
        subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', MOVIE, '-vf', 'select=eq(n\\,%d)' % k,
                        '-vsync', 'vfr', '-frames:v', '1', out], check=True)
    return out


def chain_order(c):
    """Order the cluster members along the chain (walk from an end of the bond graph)."""
    w = np.array(c['world']); q = np.array(c['charges']); n = len(q)
    dr = w[:, None, :] - w[None, :, :]; dr -= L * np.round(dr / L)
    r = np.sqrt((dr ** 2).sum(-1))
    bonds = (q[:, None] * q[None, :] < 0) & (r < BOND)
    np.fill_diagonal(bonds, False)
    deg = bonds.sum(1)
    start = int(np.argmin(deg))
    order, seen = [start], {start}
    while len(order) < n:
        nxt = [j for j in np.nonzero(bonds[order[-1]])[0] if j not in seen]
        if not nxt:
            break
        order.append(int(nxt[0])); seen.add(int(nxt[0]))
    rem = [j for j in range(n) if j not in seen]
    order += rem
    return order, int(deg.max()), r


fig, axs = plt.subplots(1, 3, figsize=(FULL_W, 2.45),
                        gridspec_kw=dict(left=0.012, right=0.995, top=0.90, bottom=0.01, wspace=0.04))
numbers = {'panels': []}
for ax, (k, size, label), pl in zip(axs, SEL, '(a) (b) (c)'.split()):
    row = rows[k]
    cl = [c for c in row['clusters'] if c['size'] == size and c['nowrap']]
    cl.sort(key=lambda c: -c['iso'])
    c = cl[0]
    pix = np.array(c['pix']); depth = np.array(c['depth'])
    cx, cy = pix.mean(0)
    x0, y0 = int(round(cx - HALF_W_PX)), int(round(cy - HALF_H_PX))
    x1, y1 = x0 + 2 * HALF_W_PX, y0 + 2 * HALF_H_PX
    png = extract(k)
    im = Image.open(png).convert('RGB').crop((x0, y0, x1, y1))
    ax.imshow(np.asarray(im), extent=(x0, x1, y1, y0), interpolation='antialiased')
    ax.set_xlim(x0, x1); ax.set_ylim(y1, y0)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True); s.set_color('#CCCCD0'); s.set_linewidth(0.5)
    # sphere radius in px at the cluster depth: r_px = 0.5 sigma * (H/2) / (depth * tan(fov/2))
    r_px = 0.5 * 360.0 / (depth.mean() * TAN)
    rad = float(np.max(np.linalg.norm(pix - pix.mean(0), axis=1))) + 1.6 * r_px
    ax.add_patch(Circle((cx, cy), rad, fill=False, ec='#222222', lw=0.9, ls=(0, (2, 2)), zorder=5))
    order, maxdeg, rmat = chain_order(c)
    seq = ' '.join('+' if c['charges'][i] > 0 else '−' for i in order)
    t = float(row['t'])
    ax.text(0.03, 0.965, 't = %.2f τ' % t, transform=ax.transAxes, ha='left', va='top', fontsize=8.5,
            bbox=dict(fc='white', ec='none', alpha=0.85, pad=1.5))
    ax.text(0.03, 0.035, '%s   %s' % (label, seq), transform=ax.transAxes, ha='left', va='bottom', fontsize=8.5,
            bbox=dict(fc='white', ec='none', alpha=0.85, pad=1.5))
    # 1 sigma scale bar at the cluster depth
    sb = 2 * r_px
    ax.plot([x1 - 12 - sb, x1 - 12], [y1 - 14, y1 - 14], color='#222222', lw=1.6, solid_capstyle='butt')
    ax.text(x1 - 12 - sb / 2, y1 - 19, '1 σ', ha='center', va='bottom', fontsize=8,
            bbox=dict(fc='white', ec='none', alpha=0.85, pad=1.0))
    ax.text(-0.0, 1.03, pl, transform=ax.transAxes, fontsize=11, fontweight='bold', va='bottom', ha='left')
    bonded = [float(rmat[order[i], order[i + 1]]) for i in range(len(order) - 1)]
    numbers['panels'].append(dict(
        panel=pl, frame_index=k, time_tau=t, label=label, cluster_size=size, charge_sequence=seq, max_degree=maxdeg,
        member_particle_ids=[c['members'][i] for i in order], bonded_distances_sigma=[round(v, 3) for v in bonded],
        nearest_other_ion_sigma=round(float(c['iso']), 2), pixel_centre=[round(float(cx), 1), round(float(cy), 1)],
        camera_depth_sigma=round(float(depth.mean()), 1), sphere_radius_px=round(float(r_px), 1),
        crop_px_x0_y0_x1_y1=[x0, y0, x1, y1], pairs_in_frame=int(row['npairs']), largest_cluster_in_frame=int(row['largest']),
        source_png=os.path.relpath(png, ROOT)))

png = savefig(fig, 'fig_pair_closeup')
print('wrote', png)

largest = np.array([r['largest'] for r in rows]); npairs = np.array([r['npairs'] for r in rows])
numbers['scan'] = dict(
    bond_criterion_sigma=BOND, n_frames=int(len(rows)),
    largest_cluster_max=int(largest.max()), largest_cluster_mean=round(float(largest.mean()), 2),
    frames_with_largest_ge4=int((largest >= 4).sum()), frames_with_largest_ge5=int((largest >= 5).sum()),
    frames_with_largest_eq6=int((largest == 6).sum()),
    first_six_ion_cluster_frames=[int(f) for f in np.nonzero(largest == 6)[0]],
    contact_pairs_per_frame_mean=round(float(npairs.mean()), 2), contact_pairs_min=int(npairs.min()), contact_pairs_max=int(npairs.max()),
    camera=dict(fov_deg=30.0, elev_deg=22.0, azimuth_deg_range=[25.0, 65.0], distance_sigma=round(float(cam.distance), 3)))
piece = {
    'file': 'fig_pair_closeup', 'width': 'full',
    'caption': ('Close-ups cut from the x-ray movie (neutrals 85 %% transparent, ions opaque; red +, blue −) at frames '
                'chosen from a cluster scan of the trajectory with the r < 1.2 σ contact criterion: (a) an isolated '
                '+/− contact pair at t = %.2f τ (nearest other ion %.1f σ away), (b) a linear four-ion chain (%s) at '
                't = %.2f τ and (c) a five-ion chain (%s) at t = %.2f τ. Dotted circles enclose the cluster; the bar gives '
                '1 σ at the depth of the cluster. Such chains are transient: clusters of four or more ions exist in %d of '
                'the 721 frames and the largest cluster seen has six ions.') % (
                    numbers['panels'][0]['time_tau'], numbers['panels'][0]['nearest_other_ion_sigma'],
                    numbers['panels'][1]['charge_sequence'], numbers['panels'][1]['time_tau'],
                    numbers['panels'][2]['charge_sequence'], numbers['panels'][2]['time_tau'],
                    numbers['scan']['frames_with_largest_ge4']),
    'numbers': numbers,
}
json.dump(piece, open(os.path.join(DATADIR, '_method_fig_pair_closeup.json'), 'w'), indent=1)
print(json.dumps(numbers, indent=1))
