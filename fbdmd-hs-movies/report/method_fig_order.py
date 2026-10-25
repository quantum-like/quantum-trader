#!/usr/bin/env python
"""fig_order: energy-error order test, max|E(t)-E(0)|/N vs dt for the four integrators.
(a) equilibrated start (notes/order_test.csv), (b) un-equilibrated start (traj/hs_charges_20p20m_order.csv)
plus the DSF variant (traj/_test/cv_confirmer_order/dsf_order.csv) for cv-phys and cv."""
import sys, os, json, csv
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, '/home/user/quantum-trader/fbdmd-hs-movies/report')
from figstyle import *  # noqa
apply_style()

ROOT = '/home/user/quantum-trader/fbdmd-hs-movies'
F_EQ = ROOT + '/notes/order_test.csv'
F_UN = ROOT + '/traj/hs_charges_20p20m_order.csv'
F_DSF = ROOT + '/traj/_test/cv_confirmer_order/dsf_order.csv'
INTEGS = ['cv-phys', 'cv', 'kdk-phys', 'naive']
MARK = {'cv-phys': 'o', 'cv': 's', 'kdk-phys': '^', 'naive': 'D'}
NICE = {'cv-phys': 'cv-phys', 'cv': 'cv (HLL)', 'kdk-phys': 'kdk-phys', 'naive': 'naive'}


def read(path):
    out = {}
    with open(path) as fh:
        for row in csv.DictReader(fh):
            out.setdefault(row['integrator'], []).append(
                dict(dt=float(row['dt']), maxdE=float(row['max_abs_dE_per_N']), end=float(row['end_drift_per_N']),
                     rms=float(row['rms_dE_per_N']), slope_csv=float(row['slope_max']), slope_end_csv=float(row['slope_end']),
                     ncoll=int(row['n_collisions']), nsub=int(row['n_substeps']), wall=float(row['wall'])))
    for k in out:
        out[k].sort(key=lambda d: d['dt'])
    return out


def fit(rows):
    x = np.log([d['dt'] for d in rows]); y = np.log([d['maxdE'] for d in rows])
    return float(np.polyfit(x, y, 1)[0])


EQ, UN, DSF = read(F_EQ), read(F_UN), (read(F_DSF) if os.path.exists(F_DSF) else None)

fig, (axa, axb) = plt.subplots(1, 2, figsize=(FULL_W, 3.15), sharey=True,
                               gridspec_kw=dict(left=0.09, right=0.99, top=0.94, bottom=0.155, wspace=0.08))
numbers = {'equilibrated_start': {}, 'unequilibrated_start': {}, 'dsf_unequilibrated_start': {}}


def refs(ax):
    """Slope fan (reference lines of slope 1 and 2) anchored in the empty lower-right corner."""
    xs = np.array([0.0075, 0.025])
    y1 = 2.2e-7 * (xs / xs[0])
    y2 = 2.2e-7 * (xs / xs[0]) ** 2
    ax.plot(xs, y1, color='#888888', ls='--', lw=0.8, zorder=1)
    ax.plot(xs, y2, color='#888888', ls='--', lw=0.8, zorder=1)
    ax.text(xs[1] * 1.04, y1[1], r'$\propto h$', color='#666666', fontsize=8, va='center', ha='left')
    ax.text(xs[1] * 1.04, y2[1], r'$\propto h^2$', color='#666666', fontsize=8, va='center', ha='left')


def panel(ax, data, key, dsf=None):
    for integ in INTEGS:
        rows = data[integ]
        s = fit(rows)
        dts = [d['dt'] for d in rows]; ys = [d['maxdE'] for d in rows]
        ax.plot(dts, ys, marker=MARK[integ], ms=4.5, color=C_INTEG[integ], lw=1.3,
                label='%s  %.2f' % (NICE[integ], s), zorder=3)
        numbers[key][integ] = dict(slope_fit=round(s, 3), slope_csv=rows[0]['slope_csv'],
                                   slope_end_drift_csv=rows[0]['slope_end_csv'],
                                   max_abs_dE_per_N_kT={str(d['dt']): d['maxdE'] for d in rows},
                                   end_drift_per_N_kT={str(d['dt']): d['end'] for d in rows})
    if dsf is not None:
        for integ in ('cv-phys', 'cv'):
            rows = dsf[integ]
            s = fit(rows)
            ax.plot([d['dt'] for d in rows], [d['maxdE'] for d in rows], marker=MARK[integ], ms=4.5,
                    mfc='white', color=C_INTEG[integ], lw=1.1, ls='--', label='%s DSF  %.2f' % (integ, s), zorder=3)
            numbers['dsf_unequilibrated_start'][integ] = dict(
                slope_fit=round(s, 3), slope_csv=rows[0]['slope_csv'],
                max_abs_dE_per_N_kT={str(d['dt']): d['maxdE'] for d in rows})
        for integ in ('kdk-phys', 'naive'):   # numbers only (not plotted, to keep the panel readable)
            rows = dsf[integ]
            numbers['dsf_unequilibrated_start'][integ] = dict(
                slope_fit=round(fit(rows), 3), slope_csv=rows[0]['slope_csv'],
                max_abs_dE_per_N_kT={str(d['dt']): d['maxdE'] for d in rows})
    refs(ax)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlim(0.002, 0.03)
    ax.set_xticks([0.0025, 0.005, 0.01, 0.02]); ax.set_xticklabels(['0.0025', '0.005', '0.01', '0.02'])
    ax.set_xticks([], minor=True)
    ax.set_xlabel('time step h (τ)')
    ax.grid(True, which='major'); ax.grid(False, which='minor')


panel(axa, EQ, 'equilibrated_start')
panel(axb, UN, 'unequilibrated_start', dsf=DSF)
axa.set_ylim(7e-8, 0.3)
axa.set_ylabel(r'max$_t\,|E(t)-E(0)|\,/\,N$  (kT)')
axa.legend(loc='upper left', ncol=2, fontsize=8, handlelength=1.8, columnspacing=0.9, borderaxespad=0.2,
           title='equilibrated start (15 τ), Ewald — log–log slope:', title_fontsize=8, alignment='left')
axb.legend(loc='upper left', ncol=2, fontsize=8, handlelength=1.8, columnspacing=0.9, borderaxespad=0.2,
           title='unequilibrated start; Ewald solid, DSF dashed:', title_fontsize=8, alignment='left')
# Ewald real-space truncation floor at the smallest step
floor = 1e-6
axb.plot([0.0021, 0.0033], [floor, floor], color='#444444', ls=':', lw=1.1, zorder=2)
axb.text(0.0021, 1.45e-7, 'Ewald truncation floor ≈ 1e−6 kT/N', fontsize=8, color='#444444', ha='left', va='center')
axb.annotate('', xy=(0.0027, floor * 0.9), xytext=(0.0027, 2.0e-7),
             arrowprops=dict(arrowstyle='-|>', color='#777777', lw=0.6, mutation_scale=7))
panel_label(axa, '(a)'); panel_label(axb, '(b)')

png = savefig(fig, 'fig_order')
print('wrote', png)

# derived numbers
eq = numbers['equilibrated_start']; un = numbers['unequilibrated_start']
numbers['ratios'] = {
    'cv_over_cvphys_at_dt0.005_equilibrated': eq['cv']['max_abs_dE_per_N_kT']['0.005'] / eq['cv-phys']['max_abs_dE_per_N_kT']['0.005'],
    'cv_over_cvphys_at_dt0.0025_equilibrated': eq['cv']['max_abs_dE_per_N_kT']['0.0025'] / eq['cv-phys']['max_abs_dE_per_N_kT']['0.0025'],
    'naive_over_cvphys_at_dt0.005_equilibrated': eq['naive']['max_abs_dE_per_N_kT']['0.005'] / eq['cv-phys']['max_abs_dE_per_N_kT']['0.005'],
    'kdkphys_over_cvphys_at_dt0.005_equilibrated': eq['kdk-phys']['max_abs_dE_per_N_kT']['0.005'] / eq['cv-phys']['max_abs_dE_per_N_kT']['0.005'],
}
numbers['test_setup'] = dict(N=1000, eta=0.35, nplus=20, nminus=20, l_B_sigma=5.0, t_run_tau=2.0, dt_frame_tau=0.02,
                             dts_tau=[0.02, 0.01, 0.005, 0.0025], production_dt_tau=0.005,
                             production_energy_drift_rel_18tau=2.3165e-05, production_energy_max_dev_rel=2.36033e-05,
                             ewald_floor_kT_per_N=1e-6, erfc_alpha_rc=7.5e-5, alpha_L=5.6, rc_sigma=5.718474607)
piece = {
    'file': 'fig_order', 'width': 'full',
    'caption': ('Energy-error order test on the production system (N = 1000, η = 0.35, 20+/20−, l_B = 5 σ): '
                'maximum |E(t) − E(0)|/N over 2 τ of NVE dynamics versus the step h for the four integrators, with '
                'the fitted log–log slopes in the legends and dashed reference lines ∝ h and ∝ h². (a) Equilibrated '
                'start (15 τ with charges on): the shipped cv-phys is second order (slope %.2f) whereas the literal HLL '
                'collision Verlet is only first order (slope %.2f) and 35× less accurate at the production step h = 0.005 τ; '
                'kdk-phys gives %.2f and the naive splitting %.2f. (b) Same test from the un-equilibrated start with Ewald '
                '(solid; slopes %.2f / %.2f / %.2f / %.2f) and with the smooth DSF Coulomb model (dashed, cv-phys %.2f, cv %.2f); '
                'the Ewald real-space truncation erfc(αr_c) ≈ 7.5×10⁻⁵ sets a floor of ≈ 10⁻⁶ kT/N at h = 0.0025 τ '
                'that flattens every Ewald curve there.') % (
                    eq['cv-phys']['slope_fit'], eq['cv']['slope_fit'], eq['kdk-phys']['slope_fit'], eq['naive']['slope_fit'],
                    un['cv-phys']['slope_fit'], un['cv']['slope_fit'], un['kdk-phys']['slope_fit'], un['naive']['slope_fit'],
                    numbers['dsf_unequilibrated_start']['cv-phys']['slope_fit'], numbers['dsf_unequilibrated_start']['cv']['slope_fit']),
    'numbers': numbers,
}
json.dump(piece, open(os.path.join(DATADIR, '_method_fig_order.json'), 'w'), indent=1)
print(json.dumps({k: {i: v['slope_fit'] for i, v in numbers[k].items()} for k in ('equilibrated_start', 'unequilibrated_start', 'dsf_unequilibrated_start')}, indent=1))
print('ratios', numbers['ratios'])
