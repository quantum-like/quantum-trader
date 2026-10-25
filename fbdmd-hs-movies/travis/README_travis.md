# Analysing the trajectories with TRAVIS

[TRAVIS](http://www.travis-analyzer.de) (Brehm & Kirchner, *J. Chem. Inf. Model.* 51, 2007 (2011); Brehm et al.,
*J. Chem. Phys.* 152, 164105 (2020)) reads the exported trajectories directly. TRAVIS was not available in the
cloud session that produced the movies (its download site and GitHub are blocked there), so the recipe below was
written from the TRAVIS manual and **not executed**; the first interactive run writes an `input.txt` that you can
replay with `travis -p <traj> -i input.txt` afterwards.

## 1. Export (done by `run_all.sh`? no — run it once)

```bash
. .venv/bin/activate
python tools/export_vmd_travis.py traj/hs_charges_20p20m.npz            # wrapped coordinates (RDF, aggregation, clusters)
python tools/export_vmd_travis.py traj/hs_charges_20p20m.npz --unwrap --outdir traj/unwrapped   # for MSD
python tools/export_vmd_travis.py traj/hs_neutral.npz
```
Files: `traj/<run>_elem.xyz` (multi-frame XYZ, element labels), `traj/<run>.lammpstrj`, `traj/<run>_frame0.pdb`.

Units: TRAVIS works in picometres and recognises molecules by covalent radii, so the reduced units are mapped to
**1 σ = 4 Å = 400 pm** (Ar = neutral hard sphere, Na = +e, Cl = −e). No bonds are detected at this scale: every
sphere is its own molecule ("Ar", "Na", "Cl"), which is exactly what the analyses below assume. Divide TRAVIS
distances by 400 pm to get σ; 1 frame = 0.025 τ (tell TRAVIS a time step of 25 fs so that 1 ps = 1 τ).

## 2. Interactive session (what to answer)

```bash
cd fbdmd-hs-movies
travis -p traj/hs_charges_20p20m_elem.xyz
```
* advanced mode: **no** (the defaults are fine).
* periodic cell: **yes**, orthogonal / cubic, edge length **4574.78 pm** (= 11.43695 σ × 400 pm; `traj/hs_charges_20p20m.json` → `L`).
* time step between frames: **25 fs** (then TRAVIS "ps" = τ).
* molecule recognition: accept the three species Ar (960), Na (20), Cl (20).
* analyses (function keys as listed by TRAVIS):
  * `rdf` — radial distribution functions. Reference Na, observed Cl; repeat for Na–Na (pool with Cl–Cl), Ar–Ar, Na–Ar.
    Max radius 2287 pm (L/2), resolution ≈ 10 pm (≈ 0.025 σ). Compare with Figure 5 of the report
    (`python tools/compare_travis.py rdf_*.csv --pair Na-Cl`).
  * `aggr` — aggregation functions (dimer lifetime / DACF). Reference Na, observed Cl, distance criterion **480 pm**
    (= 1.2 σ, the contact criterion used in the report). The intermittent lifetime should come out around 0.5 τ
    (mean) for the 1.2 σ contact criterion; report Table 5.
  * `cluster` — hierarchical cluster analysis of the ions (Na + Cl, distance criterion 480 pm) for cluster-size
    distributions (report Figure 6c: 62 / 27 / 8 / 2 % of ions in clusters of size 1 / 2 / 3 / 4).
  * `msd` — mean-squared displacement: **use the unwrapped export** (`traj/unwrapped/..._elem.xyz`), observe Ar and
    Na/Cl separately; D ≈ MSD/6t from the 1–4 τ window (report Figure 7a).
  * `vacf` is not available: the trajectory stores positions only (velocities change at every collision and were
    not written; TRAVIS could differentiate positions, but the frame spacing of 0.025 τ is ≈ 1/3 of a collision time,
    so the result would not be a true VACF).
* processing: all 721 frames, stride 1.

TRAVIS writes `rdf_*.csv`, `dacf_*.csv`/`lifetime_*`, `cluster_*.csv`, `msd_*.csv` (semicolon separated, `#` header)
into the working directory; `tools/compare_travis.py` overlays an RDF csv on the report's own g(r).

## 3. Why TRAVIS and the report's numbers should agree

The report's statistics (notes/analysis_hs_charges_20p20m.txt, report/figures/fig_pairing.png, fig_structure.png)
were computed with plain numpy from the same frames; the independent auditors re-derived them. TRAVIS's RDF and
aggregation functions implement the same definitions (minimum-image pair distances, intermittent contact criterion),
so differences beyond the binning should be reported as a bug in either tool. Known conventions to watch:
TRAVIS normalises g(r) with the number density of the *observed* species; lifetimes from DACF are the integral of the
autocorrelation (τ_int), not the mean episode length listed in Table 5 (0.52 τ mean, 0.35 τ median) — both are in
the report for comparison.
