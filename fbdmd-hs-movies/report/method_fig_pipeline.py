#!/usr/bin/env python
"""fig_pipeline: block diagram of the fbdmd-hs-movies pipeline, drawn with matplotlib patches only
(rounded boxes + arrows, no raster).  Writes report/figures/fig_pipeline.{png,svg} and a manifest piece."""
import sys, os, json
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

sys.path.insert(0, '/home/user/quantum-trader/fbdmd-hs-movies/report')
from figstyle import *  # noqa
apply_style()
plt.rcParams['axes.grid'] = False

W, H = FULL_W, 2.6
fig = plt.figure(figsize=(W, H))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W); ax.set_ylim(0, H); ax.set_aspect('equal'); ax.axis('off')

FILL = {'sim': C_ACCENT, 'simlight': '#DCE6F1', 'data': '#ECECF0', 'render': '#F8E3C4', 'bar': '#F4F4F7'}
EDGE = {'sim': C_ACCENT, 'simlight': C_ACCENT, 'data': C_NEUTRAL, 'render': '#C9781B', 'bar': '#5A5A64'}
TEXT = {'sim': 'white', 'simlight': C_ACCENT, 'data': '#222222', 'render': '#3A2A10', 'bar': '#222222'}
_checks = []   # (text artist, available width in inches) for a legibility/overflow check


def box(x0, y0, w, h, kind, title, body='', ls='-', title_size=8.5, body_size=8):
    ax.add_patch(FancyBboxPatch((x0, y0), w, h, boxstyle='round,pad=0,rounding_size=0.07',
                                fc=FILL[kind], ec=EDGE[kind], lw=0.9, ls=ls, zorder=2))
    xc = x0 + w / 2
    if body:
        nb = body.count('\n') + 1
        lh = body_size / 72 * 1.25
        total = title_size / 72 * 1.35 + nb * lh
        ytop = y0 + h / 2 + total / 2
        t = ax.text(xc, ytop, title, ha='center', va='top', fontsize=title_size, fontweight='bold',
                    color=TEXT[kind], zorder=3)
        b = ax.text(xc, ytop - title_size / 72 * 1.35, body, ha='center', va='top', fontsize=body_size,
                    color=TEXT[kind], linespacing=1.25, zorder=3)
        _checks.extend([(t, w), (b, w)])
    else:
        t = ax.text(xc, y0 + h / 2, title, ha='center', va='center', fontsize=title_size, fontweight='bold',
                    color=TEXT[kind], zorder=3)
        _checks.append((t, w))


def arrow(p0, p1, color='#444444', ls='-', lw=0.9, head=True):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle='-|>' if head else '-', mutation_scale=9, color=color,
                                 lw=lw, linestyle=ls, shrinkA=0, shrinkB=0, zorder=4))


# ---- orchestration bar
bx0, by0, bw, bh = 0.12, 2.08, 6.96, 0.46
ax.add_patch(FancyBboxPatch((bx0, by0), bw, bh, boxstyle='round,pad=0,rounding_size=0.07',
                            fc=FILL['bar'], ec=EDGE['bar'], lw=0.9, zorder=2))
t = ax.text(bx0 + 0.12, by0 + bh / 2, 'run_all.sh', ha='left', va='center', fontsize=9.5, fontweight='bold',
            color=TEXT['bar'], zorder=3)
_checks.append((t, 1.0))
bar_body = ('one-shot pipeline:  venv → make → selftest → simulate → analyze → render → NOTES.md\n'
            'knobs (env):  N · η · l_B · SEED · INTEGRATOR (cv-phys | cv | kdk-phys | naive) · DT · T_RUN · DT_FRAME\n'
            'QUALITY (final | draft) · RENDERER (ovito | numpy) · WIDTH × HEIGHT · FPS · ORBIT · XRAY · ORDER_TEST')
b = ax.text(bx0 + 0.95, by0 + bh / 2, bar_body, ha='left', va='center', fontsize=8, color=TEXT['bar'],
            linespacing=1.3, zorder=3)
_checks.append((b, bw - 0.97))

# ---- simulation column
box(0.12, 0.72, 1.40, 1.14, 'sim', 'sim/hsdmd.cpp',
    'C++17, single file\nEDMD hard cores\n+ collision Verlet\n+ Ewald / DSF Coulomb\n(run_sim.py wrapper)')
box(0.12, 0.14, 1.40, 0.42, 'simlight', 'sim/order_test.py', '→ notes/order_test.csv', ls='--')
arrow((0.82, 0.72), (0.82, 0.56), color=C_ACCENT, ls=':', lw=0.9)

# ---- trajectory data
box(1.92, 0.72, 1.40, 1.14, 'data', 'traj/<run>',
    '.xyz   ext-XYZ (OVITO)\n.npz   float32 cache\n.json  metadata\n_diag.csv  per frame')
arrow((1.52, 1.29), (1.92, 1.29))

# ---- analysis branch (upper)
box(3.72, 1.45, 1.50, 0.50, 'sim', 'sim/analyze.py', 'overlaps, energy drift,\npairing & cluster stats')
box(5.42, 1.45, 1.66, 0.50, 'data', 'notes/ + NOTES.md', 'analysis_*.txt\nmake_notes.py → NOTES.md')
arrow((3.32, 1.50), (3.72, 1.72))
arrow((5.22, 1.70), (5.42, 1.70))

# ---- render branch (lower)
box(3.72, 0.50, 1.60, 0.70, 'render', 'render/render_ovito.py',
    'OVITO Tachyon ray tracer\nAO 16 · shadows · 4× AA\nfallback: render_numpy.py')
box(5.55, 0.50, 0.68, 0.70, 'data', 'frames/', '*.png\n1280×720')
box(6.46, 0.50, 0.62, 0.70, 'data', 'movies/', '*.mp4\nH.264')
arrow((3.32, 1.05), (3.72, 0.87))
arrow((5.32, 0.85), (5.55, 0.85))
arrow((6.23, 0.85), (6.46, 0.85))
t = ax.text(6.345, 0.44, 'ffmpeg (libx264, 30 fps)', ha='center', va='top', fontsize=8, color='#333333')
_checks.append((t, 1.6))

# ---- bar connectors (orchestration)
for x, ytop in ((0.82, 1.86), (4.47, 1.95), (6.25, 1.95)):
    ax.plot([x, x], [by0, ytop], color=EDGE['bar'], lw=0.8, ls=(0, (1, 2)), zorder=1)

# ---- overflow check: every text must fit its box (in inches)
fig.canvas.draw()
ren = fig.canvas.get_renderer()
problems = []
for t, wavail in _checks:
    bb = t.get_window_extent(ren)
    w_in = bb.width / fig.dpi
    if w_in > wavail - 0.05:
        problems.append((t.get_text().split('\n')[0], round(w_in, 2), wavail))
for p in problems:
    print('OVERFLOW:', p)
png = savefig(fig, 'fig_pipeline')
print('wrote', png)

piece = {
    'file': 'fig_pipeline', 'width': 'full',
    'caption': ('Pipeline of the project: run_all.sh orchestrates the C++17 simulator (event-driven hard-sphere MD '
                'with collision Verlet and Ewald/DSF Coulomb), which writes the trajectory as extended XYZ plus '
                'an npz cache, JSON metadata and a per-frame diagnostics CSV; sim/analyze.py scans every frame for '
                'overlaps and computes energy drift and pairing statistics, while render_ovito.py (Tachyon ray '
                'tracer, numpy fallback) produces 721 PNG frames per movie that ffmpeg encodes to H.264 mp4.'),
    'numbers': {
        'figure_size_in': [W, H],
        'frames_per_movie': 721, 'frame_size_px': [1280, 720], 'fps': 30,
        'run_all_knobs': ['N', 'ETA', 'BJERRUM', 'SEED', 'INTEGRATOR', 'DT', 'T_RUN', 'DT_FRAME', 'T_EQ_N', 'T_EQ_C',
                          'QUALITY', 'RENDERER', 'WIDTH', 'HEIGHT', 'FPS', 'ORBIT', 'XRAY', 'ORDER_TEST', 'SELFTEST',
                          'FORCE', 'PYTHON'],
        'text_overflow_problems': problems,
    },
}
json.dump(piece, open(os.path.join(DATADIR, '_method_fig_pipeline.json'), 'w'), indent=1)
