#!/usr/bin/env python
"""fig_pair_closeup: close-ups of a +/- contact pair and of ion chains in the charged production run.

Frames were chosen with a cluster scan of traj/hs_charges_20p20m.npz (traj/_test/report/cluster_scan.json, bond
criterion r(+-) < 1.2 sigma as in sim/analyze.py) plus a projection check (render/common.py Camera, read-only import):
clusters at a camera depth < 42 sigma, no other ion inside the dotted circle in projection, few neutrals in front.
The three frames are re-rendered with the shipped renderer at 3x the movie resolution (3840 x 2160, x-ray style,
final quality, no overlay) into traj/_test/report/closeup_frames/ so that the 900 x 792 px crops are sharp; the
camera, colours and transparency are those of movies/hs_charges_20p20m_xray.mp4 (same orbit azimuth per frame).
"""
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
TRAJ = ROOT + '/traj/hs_charges_20p20m.npz'
FRAMES3X = ROOT + '/traj/_test/report/closeup_frames'      # scratch (gitignored)
BOND = 1.2
SCALE = 3                                                  # render at 3x the 1280x720 movie resolution
W0, H0 = 1280, 720
# (frame index K, cluster size, short label, approximate pixel centre in the 1280x720 frame used to pick the cluster)
SEL = [(651, 2, '+/− contact pair', (706, 502)), (390, 4, '4-ion chain', (778, 382)), (369, 5, '5-ion chain', (805, 241))]
HALF_W_PX, HALF_H_PX = 150, 132          # crop half-sizes in 720p pixels (-> 900 x 792 px at 3x)

rows = json.load(open(SCAN))
d = np.load(TRAJ, allow_pickle=False)
L = float(d['L']); times = d['times']; F = len(times)
pos = d['pos'].astype(np.float64); types = d['types']
ions = np.nonzero(types > 1)[0]; neut = np.nonzero(types == 1)[0]
cam = Camera(L, W0, H0, nout=F)
TAN = np.tan(np.radians(15.0))      # vertical half-FOV (30 deg / 2)


def render3x(k):
    """Frame k of the x-ray movie at 3x resolution via the shipped renderer (resumable: skipped if present)."""
    out = os.path.join(FRAMES3X, 'frame_%05d.png' % k)
    if not os.path.exists(out):
        os.makedirs(FRAMES3X, exist_ok=True)
        subprocess.run([sys.executable, ROOT + '/render/render_ovito.py', '--traj', TRAJ, '--name', 'closeup3x',
                        '--frames-dir', FRAMES3X, '--width', str(W0 * SCALE), '--height', str(H0 * SCALE),
                        '--style', 'xray', '--quality', 'final', '--start', str(k), '--end', str(k),
                        '--no-overlay', '--no-encode', '--keep-frames'], check=True, cwd=ROOT)
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
for ax, (k, size, label, (gx, gy)), pl in zip(axs, SEL, '(a) (b) (c)'.split()):
    row = rows[k]
    cl = [c for c in row['clusters'] if c['size'] == size and c['nowrap']]
    c = min(cl, key=lambda c_: np.hypot(*(np.mean(c_['pix'], 0) - [gx, gy])))
    mem = np.array(c['members'])
    pose = cam.pose(k)
    pix_all, dep_all = cam.project(pose, pos[k])
    pix, depth = pix_all[mem], dep_all[mem]
    cx, cy = pix.mean(0)
    # sphere radius in px at the cluster depth: r_px = 0.5 sigma * (H/2) / (depth * tan(fov/2))
    r_px = 0.5 * (H0 / 2.0) / (depth.mean() * TAN)
    rad = float(np.max(np.linalg.norm(pix - pix.mean(0), axis=1))) + 1.6 * r_px
    # projection checks: other ions and neutrals inside the circle
    others = np.setdiff1d(ions, mem)
    d_other = np.linalg.norm(pix_all[others] - [cx, cy], axis=1)
    n_ion_in_circle = int((d_other < rad + r_px).sum())
    d_neut = np.linalg.norm(pix_all[neut] - [cx, cy], axis=1)
    n_neut_front = int(((d_neut < rad) & (dep_all[neut] < depth.mean() - 0.5)).sum())
    n_neut_behind = int(((d_neut < rad) & (dep_all[neut] > depth.mean() + 0.5)).sum())
    # crop (3x pixels)
    x0, y0 = int(round(cx - HALF_W_PX)), int(round(cy - HALF_H_PX))
    x1, y1 = x0 + 2 * HALF_W_PX, y0 + 2 * HALF_H_PX
    png = render3x(k)
    im = Image.open(png).convert('RGB').crop((x0 * SCALE, y0 * SCALE, x1 * SCALE, y1 * SCALE))
    ax.imshow(np.asarray(im), extent=(x0, x1, y1, y0), interpolation='antialiased')   # axes in 720p pixel units
    ax.set_xlim(x0, x1); ax.set_ylim(y1, y0)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True); s.set_color('#CCCCD0'); s.set_linewidth(0.5)
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
    idx0 = [int(c['members'][i]) for i in order]
    numbers['panels'].append(dict(
        panel=pl, frame_index=k, time_tau=t, label=label, cluster_size=size, charge_sequence=seq, max_degree=maxdeg,
        member_particle_indices=idx0, member_xyz_ids=[i + 1 for i in idx0],
        bonded_distances_sigma=[round(v, 3) for v in bonded],
        nearest_other_ion_sigma=round(float(c['iso']), 2), other_ions_inside_circle_in_projection=n_ion_in_circle,
        neutrals_in_front_inside_circle=n_neut_front, neutrals_behind_inside_circle=n_neut_behind,
        pixel_centre_720p=[round(float(cx), 1), round(float(cy), 1)],
        camera_depth_sigma=round(float(depth.mean()), 1), sphere_radius_px_720p=round(float(r_px), 1),
        circle_radius_px_720p=round(float(rad), 1),
        crop_px_720p_x0_y0_x1_y1=[x0, y0, x1, y1], crop_size_px_rendered=[2 * HALF_W_PX * SCALE, 2 * HALF_H_PX * SCALE],
        pairs_in_frame=int(row['npairs']), largest_cluster_in_frame=int(row['largest']),
        source_png=os.path.relpath(png, ROOT)))

png = savefig(fig, 'fig_pair_closeup')
print('wrote', png)

largest = np.array([r['largest'] for r in rows]); npairs = np.array([r['npairs'] for r in rows])
panel_w_in = FULL_W * (0.995 - 0.012 - 2 * 0.04 * 0.33) / 3.0
numbers['scan'] = dict(
    bond_criterion_sigma=BOND, n_frames=int(len(rows)),
    largest_cluster_max=int(largest.max()), largest_cluster_mean=round(float(largest.mean()), 2),
    frames_with_largest_ge4=int((largest >= 4).sum()), frames_with_largest_ge5=int((largest >= 5).sum()),
    frames_with_largest_eq6=int((largest == 6).sum()),
    first_six_ion_cluster_frames=[int(f) for f in np.nonzero(largest == 6)[0]],
    six_ion_clusters_note='all 8 six-ion clusters (frames 527-535) are branched (max degree 3) and sit 48 sigma deep behind ~115 neutrals; the clean linear 5-ion chain of frames 368-370 is shown instead',
    contact_pairs_per_frame_mean=round(float(npairs.mean()), 2), contact_pairs_min=int(npairs.min()), contact_pairs_max=int(npairs.max()),
    camera=dict(fov_deg=30.0, elev_deg=22.0, azimuth_deg_range=[25.0, 65.0], distance_sigma=round(float(cam.distance), 3)))
numbers['render'] = dict(
    command='render/render_ovito.py --style xray --quality final --width 3840 --height 2160 --no-overlay --start K --end K',
    resolution_px=[W0 * SCALE, H0 * SCALE], scale_vs_movie=SCALE, crop_px=[2 * HALF_W_PX * SCALE, 2 * HALF_H_PX * SCALE],
    panel_width_in=round(panel_w_in, 2), effective_dpi=round(2 * HALF_W_PX * SCALE / panel_w_in), seconds_per_frame_render=12.0,
    id_convention='member_particle_indices are 0-based npz/array indices (FORMAT.md order); member_xyz_ids = index + 1 are the 1-based id column of the extended XYZ')
piece = {
    'file': 'fig_pair_closeup', 'width': 'full',
    'caption': ('Close-ups of the charged production run in the x-ray style of the movie (neutrals 85 %% transparent, ions '
                'opaque; red +, blue −), re-rendered at 3× the movie resolution for the three frames chosen from a cluster scan '
                'with the r < 1.2 σ contact criterion: (a) an isolated +/− contact pair at the front face of the box at t = %.2f τ '
                '(nearest other ion %.1f σ away, no other ion inside the circle in projection, no neutral in front of it), '
                '(b) a linear four-ion chain (%s) at t = %.2f τ and (c) a linear five-ion chain (%s) at t = %.2f τ, %.0f σ from the '
                'camera. Dotted circles enclose the cluster; the bar gives 1 σ at the depth of the cluster. Such chains are '
                'transient: clusters of four or more ions exist in %d of the 721 frames and the largest cluster seen has six ions.') % (
                    numbers['panels'][0]['time_tau'], numbers['panels'][0]['nearest_other_ion_sigma'],
                    numbers['panels'][1]['charge_sequence'], numbers['panels'][1]['time_tau'],
                    numbers['panels'][2]['charge_sequence'], numbers['panels'][2]['time_tau'], numbers['panels'][2]['camera_depth_sigma'],
                    numbers['scan']['frames_with_largest_ge4']),
    'numbers': numbers,
}
json.dump(piece, open(os.path.join(DATADIR, '_method_fig_pair_closeup.json'), 'w'), indent=1)
print(json.dumps(numbers['panels'], indent=1))
