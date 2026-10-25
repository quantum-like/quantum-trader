#!/usr/bin/env python
"""fig_cv_scheme: (a) HLL potential splitting for a +/- Coulomb pair (l_B = 5, q1 = 1, q2 = 1.5),
(b) the corresponding pair forces, (c) timeline schematic of one collision-Verlet step."""
import sys, os, json
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, '/home/user/quantum-trader/fbdmd-hs-movies/report')
from figstyle import *  # noqa
apply_style()

lB, q1, q2 = 5.0, 1.0, 1.5
phic = lambda r: -lB / r              # +/- pair: q_i q_j = -1
dphic = lambda r: lB / r ** 2         # d phi_c / dr
A3 = dphic(q2) / (3 * (q2 - q1) ** 2)
A2 = -3 * q1 * A3
A1 = 3 * q1 ** 2 * A3
A0 = phic(q2) - (A1 * q2 + A2 * q2 ** 2 + A3 * q2 ** 3)
P = lambda r: A0 + A1 * r + A2 * r ** 2 + A3 * r ** 3
dP = lambda r: A1 + 2 * A2 * r + 3 * A3 * r ** 2
Pq1 = float(P(q1))


def phi2(r):
    return np.where(r < q1, Pq1, np.where(r < q2, P(r), phic(r)))


def phi1(r):
    return phic(r) - phi2(r)


def F2(r):  # F = -dphi/dr (radial; negative = attractive)
    return np.where(r < q1, 0.0, np.where(r < q2, -dP(r), -dphic(r)))


def Fc(r):
    return -dphic(r)


def F1(r):
    return Fc(r) - F2(r)


# sanity checks (same as the simulator self-test)
rr = np.linspace(0.8, 3.0, 2001)
assert np.allclose(phi1(rr) + phi2(rr), phic(rr))
assert abs(dP(q1)) < 1e-12 and abs(2 * A2 + 6 * A3 * q1) < 1e-12 and abs(dP(q2) - dphic(q2)) < 1e-12
assert abs(P(q2) - phic(q2)) < 1e-12

C_PHIC, C_PHI2, C_PHI1 = '#222222', C_ACCENT, '#D97A00'

fig = plt.figure(figsize=(FULL_W, 3.0))
gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.95], wspace=0.48, left=0.075, right=0.99, top=0.90, bottom=0.17)
axa, axb, axc = [fig.add_subplot(gs[0, i]) for i in range(3)]

# ---------------- (a) potentials
r = np.linspace(0.8, 3.0, 1101)
for a in (axa, axb):
    a.axvspan(0.8, 1.0, color='#E4E4E8', lw=0, zorder=0)
    for q in (q1, q2):
        a.axvline(q, color='#999999', ls=':', lw=0.8, zorder=1)
    a.set_xlim(0.8, 3.0)
    a.set_xlabel('r (σ)')
axa.plot(r, phic(r), color=C_PHIC, ls='--', lw=1.2, label=r'$\varphi_\mathrm{c} = -l_\mathrm{B}/r$')
axa.plot(r, phi2(r), color=C_PHI2, lw=1.7, label=r'$\varphi_2$ (long range)')
axa.plot(r, phi1(r), color=C_PHI1, lw=1.7, label=r'$\varphi_1 = \varphi_\mathrm{c} - \varphi_2$')
axa.set_ylim(-7.5, 0.8)
axa.set_ylabel('pair potential (kT)')
axa.text(0.9, -7.15, 'hard\ncore', ha='center', va='bottom', fontsize=8, color='#555555')
axa.text(q1, 0.9, r'$q_1$', ha='center', va='bottom', fontsize=8)
axa.text(q2, 0.9, r'$q_2$', ha='center', va='bottom', fontsize=8)
axa.annotate(r'$P(q_1) = %.2f\ kT$' % Pq1, xy=(0.95, Pq1), xytext=(1.32, -4.75), fontsize=8, color=C_PHI2,
             ha='left', va='center', arrowprops=dict(arrowstyle='-', color=C_PHI2, lw=0.7, shrinkB=2))
axa.legend(loc='lower right', fontsize=8, handlelength=1.6, borderaxespad=0.2)
panel_label(axa, '(a)')

# ---------------- (b) forces
axb.plot(r, Fc(r), color=C_PHIC, ls='--', lw=1.2, label=r'$F_\mathrm{c}$')
axb.plot(r, F2(r), color=C_PHI2, lw=1.7, label=r'$F_2$')
axb.plot(r, F1(r), color=C_PHI1, lw=1.7, label=r'$F_1$')
axb.set_ylim(-8.3, 1.0)
axb.set_ylabel(r'pair force $-\mathrm{d}\varphi/\mathrm{d}r$ (kT/σ)')
axb.text(0.9, -7.95, 'hard\ncore', ha='center', va='bottom', fontsize=8, color='#555555')
axb.text(q1, 1.1, r'$q_1$', ha='center', va='bottom', fontsize=8)
axb.text(q2, 1.1, r'$q_2$', ha='center', va='bottom', fontsize=8)
axb.annotate('$F_2 = 0$ and $F_2\' = 0$\nat contact', xy=(1.0, 0.0), xytext=(1.6, -3.2), fontsize=8, color=C_PHI2,
             ha='left', va='center', arrowprops=dict(arrowstyle='-|>', color=C_PHI2, lw=0.7, mutation_scale=8))
axb.annotate('$F_1(q_2) = 0$', xy=(q2, 0.0), xytext=(1.95, -1.3), fontsize=8, color=C_PHI1, ha='left', va='center',
             arrowprops=dict(arrowstyle='-|>', color=C_PHI1, lw=0.7, mutation_scale=8))
axb.legend(loc='lower right', fontsize=8, handlelength=1.6, borderaxespad=0.2)
panel_label(axb, '(b)')

# ---------------- (c) one step of collision Verlet
ax = axc
ax.set_xlim(0, 1); ax.set_ylim(-0.19, 1); ax.axis('off')
x0, x1, yt = 0.08, 0.86, 0.17
taus = [0.30, 0.54, 0.72]
AX = '#333333'
ax.annotate('', xy=(x1 + 0.06, yt), xytext=(x0 - 0.04, yt),
            arrowprops=dict(arrowstyle='-|>', lw=1.0, color=AX, mutation_scale=9))
for x, lab in ((x0, '$t$'), (x1, '$t+h$')):
    ax.plot([x, x], [yt - 0.03, yt + 0.03], color=AX, lw=1.2)
    ax.text(x, yt - 0.04, lab, ha='center', va='top', fontsize=8.5)
for i, tx in enumerate(taus):
    ax.plot([tx, tx], [yt - 0.02, yt + 0.02], color=AX, lw=1.0)
    ax.text(tx, yt - 0.04, r'$\tau_%d$' % (i + 1), ha='center', va='top', fontsize=8.5)
# sub-step brackets
edges = [x0] + taus + [x1]
yb = yt - 0.115
for a_, b_ in zip(edges[:-1], edges[1:]):
    ax.plot([a_ + 0.006, b_ - 0.006], [yb, yb], color='#777777', lw=0.8)
    ax.plot([a_ + 0.006, a_ + 0.006], [yb, yb + 0.015], color='#777777', lw=0.8)
    ax.plot([b_ - 0.006, b_ - 0.006], [yb, yb + 0.015], color='#777777', lw=0.8)
ax.text((x0 + x1) / 2, yb - 0.02, 'one velocity-Verlet sub-step with $F_1$ per interval',
        ha='center', va='top', fontsize=8, color='#444444')
ax.text((x0 + x1) / 2, yb - 0.095, 'next collision: smallest $\\tau>0$ with\n'
        r'$|\Delta q+\Delta v\tau+\frac{1}{2}\Delta a\tau^2|^2=\sigma^2$',
        ha='center', va='top', fontsize=8, color='#222222', linespacing=1.2)
# outer half kicks with F2
for x in (x0, x1):
    ax.annotate('', xy=(x, yt + 0.13), xytext=(x, yt + 0.03),
                arrowprops=dict(arrowstyle='-|>', lw=1.6, color=C_ACCENT, mutation_scale=10))
    ax.text(x, yt + 0.145, '½h·F₂ kick', ha='center', va='bottom', fontsize=8, color=C_ACCENT)


def para(xa, xb, ya, va, acc, n=60):
    x = np.linspace(xa, xb, n); s = x - xa
    return x, ya + va * s + 0.5 * acc * s ** 2


# + ion (accelerated: parabolic arcs between collisions)
red = [(x0, taus[0], 0.50, 0.30, 2.0)]
y = red[0][2] + red[0][3] * (taus[0] - x0) + 0.5 * red[0][4] * (taus[0] - x0) ** 2
red.append((taus[0], taus[1], y, -0.60, 2.0)); s = taus[1] - taus[0]; y = y - 0.60 * s + 1.0 * s ** 2
y_pm = y
red.append((taus[1], taus[2], y, -0.35, 2.0)); s = taus[2] - taus[1]; y = y - 0.35 * s + 1.0 * s ** 2
red.append((taus[2], x1, y, 0.01, 2.0))
# - ion
blue = [(x0, taus[0], 0.77, -0.30, -2.0)]
s = taus[0] - x0; yb1 = 0.77 - 0.30 * s - 1.0 * s ** 2
s = taus[1] - taus[0]; vb = (y_pm - yb1 + 0.75 * s ** 2) / s
blue.append((taus[0], taus[1], yb1, vb, -1.5))
blue.append((taus[1], taus[2], y_pm, 0.35, -2.0)); s = taus[2] - taus[1]; yb3 = y_pm + 0.35 * s - 1.0 * s ** 2
blue.append((taus[2], x1, yb3, 0.10, -1.0))
for segs, col in ((red, C_PLUS), (blue, C_MINUS)):
    for (xa, xb, ya, va, acc) in segs:
        xx, yy = para(xa, xb, ya, va, acc)
        ax.plot(xx, yy, color=col, lw=1.8, solid_capstyle='round', zorder=3)
y_r1 = red[1][2]
# neutrals (ballistic, straight)
ax.plot([x0, taus[0], 0.52], [0.70, y_r1, 0.40], color=C_NEUTRAL, lw=1.6, zorder=2)
ax.plot([0.60, taus[2], x1], [0.83, yb3, 0.76], color=C_NEUTRAL, lw=1.6, zorder=2)
# collision markers + droplines
for tx, yy in ((taus[0], y_r1), (taus[1], y_pm), (taus[2], yb3)):
    ax.plot([tx, tx], [yt + 0.02, yy], color='#999999', lw=0.6, ls=':', zorder=1)
    ax.plot(tx, yy, 'o', ms=5.5, mfc='white', mec='#222222', mew=1.0, zorder=5)
# labels
ax.text(x1 + 0.015, red[3][2] + 0.0014 * 0 + 0.021 - 0.02, '+ ion', color=C_PLUS, fontsize=8, ha='left', va='top')
ax.text(x1 + 0.015, yb3 + 0.03, '− ion', color=C_MINUS, fontsize=8, ha='left', va='bottom')
ax.text(0.875, 0.76, 'neutral', color='#666666', fontsize=8, ha='left', va='center')
ax.text(0.10, 0.47, r'$q + v\tau + \frac{1}{2}a\tau^2$', color=C_PLUS, fontsize=8, ha='left', va='top')
ax.text(0.535, 0.40, 'ballistic', color='#666666', fontsize=8, ha='left', va='center')
# texts: quartic collision condition (left) and the cv / cv-phys collision rule (right)
ax.text(0.0, 1.0, 'one step $h$', ha='left', va='top', fontsize=8.5, color='#444444')
ax.text(1.0, 1.0, 'velocity reflected at $\\tau_i$:\ncv: drift $v$\ncv-phys: $v + F_2(\\tau_i - h/2)$',
        ha='right', va='top', fontsize=8, linespacing=1.2)
panel_label(axc, '(c)')

png = savefig(fig, 'fig_cv_scheme')
print('wrote', png)

piece = {
    'file': 'fig_cv_scheme', 'width': 'full',
    'caption': ('Collision Verlet (Houndonougbo, Laird & Leimkuhler 2000) as implemented. (a) The +/− Coulomb pair '
                'potential φc = −l_B/r (l_B = 5 σ) is split into a long-range part φ₂ that follows the Hermite '
                'cubic P(r) on [q₁, q₂] = [1, 1.5] σ and is flat at P(q₁) = −3.70 kT inside q₁, and a short-range '
                'part φ₁ = φc − φ₂ that vanishes beyond q₂; the hard core r < σ is shaded. (b) The corresponding '
                'pair forces: F₂ and its slope vanish at contact, F₁ carries the full contact force −5 kT/σ and goes '
                'to zero continuously at q₂. (c) One step h: outer half-kicks with F₂, exact hard-sphere propagation in '
                'between in which every hard collision (anywhere in the system) cuts the step into velocity-Verlet '
                'sub-steps with F₁; accelerated ions follow parabolic paths and their collision times are the smallest '
                'positive root of the quartic |Δq + Δvτ + ½Δaτ²|² = σ². The shipped cv-phys variant '
                'reflects the physical velocity v + F₂(τᵢ − h/2) at each collision instead of the drift velocity.'),
    'numbers': {
        'l_B_sigma': lB, 'q1_sigma': q1, 'q2_sigma': q2,
        'A0_kT': A0, 'A1_kT_per_sigma': A1, 'A2_kT_per_sigma2': A2, 'A3_kT_per_sigma3': A3,
        'P_q1_kT': Pq1, 'phi_c_q2_kT': float(phic(q2)), 'dphi_c_dr_q2_kT_per_sigma': float(dphic(q2)),
        'phi_c_contact_kT': float(phic(1.0)), 'phi1_contact_kT': float(phic(1.0) - Pq1),
        'F_c_contact_kT_per_sigma': float(Fc(1.0)), 'F1_contact_kT_per_sigma': float(F1(np.array([1.0]))[0]),
        'F2_contact_kT_per_sigma': 0.0, 'F_c_q2_kT_per_sigma': float(Fc(q2)),
        'plotted_r_range_sigma': [0.8, 3.0],
        'step_h_tau': 0.005, 'mean_collision_interval_system_tau': 1.4e-4,
    },
}
json.dump(piece, open(os.path.join(DATADIR, '_method_fig_cv_scheme.json'), 'w'), indent=1)
print(json.dumps(piece['numbers'], indent=1))
