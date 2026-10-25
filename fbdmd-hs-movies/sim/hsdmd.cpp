// hsdmd.cpp -- event-driven hard-sphere molecular dynamics (EDMD) with an
// optional restricted-primitive-model (RPM) charged subset integrated by
// collision Verlet (Houndonougbo, Laird, Leimkuhler, Mol. Phys. 98, 309, 2000;
// "HLL"): the Coulomb pair potential is split into a short-range part phi_1
// (zero beyond q2, Hermite-cubic matched) and a long-range part phi_2 (flat for
// r < q1 = sigma, i.e. zero force at contact).  Outer half kicks with F_2 =
// F_Ewald - F_1 enclose an exact propagation of T + V_hs + V_1: velocity-Verlet
// sub-steps with F_1 between consecutive hard collisions, quadratic paths for
// the ions with F_1 != 0, quartic collision times (HLL Table 1, eqs. 28-29).
// The older kick-drift-kick schemes (--integrator kdk-phys | naive) are kept.
//
// Reduced units: sigma = 1, m = 1, kT = 1, tau = sigma*sqrt(m/kT).
// Coulomb:  U = bjerrum * sum_{i<j} q_i q_j phi(r_ij)   (q = +-1),
//           phi = Ewald (tin-foil) or Fennell-Gezelter damped shifted force.
//
// Output (see ../FORMAT.md): extended XYZ frames at uniform times (ballistic
// free flight evaluated exactly at the frame time), a diagnostics CSV with one
// row per frame, and a JSON summary.  Single file, C++17 standard library only.
//
// Build:  c++ -O3 -std=c++17 -Wall -Wextra -o hsdmd hsdmd.cpp
// Usage:  ./hsdmd --help

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>
#include <string>
#include <vector>

namespace {

constexpr double SIGMA = 1.0;
constexpr double SIGMA2 = SIGMA * SIGMA;
constexpr double PI = 3.14159265358979323846;
constexpr double KT = 1.0;
const char* HSDMD_VERSION = "1.2";

struct Vec3 {
  double x = 0, y = 0, z = 0;
};
inline Vec3 operator+(Vec3 a, Vec3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
inline Vec3 operator-(Vec3 a, Vec3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
inline Vec3 operator*(Vec3 a, double s) { return {a.x * s, a.y * s, a.z * s}; }
inline Vec3& operator+=(Vec3& a, Vec3 b) { a.x += b.x; a.y += b.y; a.z += b.z; return a; }
inline Vec3& operator-=(Vec3& a, Vec3 b) { a.x -= b.x; a.y -= b.y; a.z -= b.z; return a; }
inline double dot(Vec3 a, Vec3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
inline double& comp(Vec3& a, int d) { return d == 0 ? a.x : (d == 1 ? a.y : a.z); }

double wall_now() {
  using namespace std::chrono;
  static const auto t0 = steady_clock::now();
  return duration<double>(steady_clock::now() - t0).count();
}

// ---------------------------------------------------------------------------
// Coulomb interaction for the ion subset (Ewald or DSF)
// ---------------------------------------------------------------------------
struct Coulomb {
  enum Mode { EWALD = 0, DSF = 1 };
  Mode mode = EWALD;
  double L = 1, V = 1, bjerrum = 1;
  double alpha = 1, rc = 0.5;
  int kmax = 6;
  // Ewald reciprocal vectors (half space, factor 2 applied in the sums)
  struct KVec { int nx, ny, nz; double kx, ky, kz, w; };
  std::vector<KVec> kvecs;
  // DSF shifts
  double dsf_shiftU = 0, dsf_shiftF = 0, dsf_self = 0;

  void setup(Mode m, double boxL, double lB) {
    mode = m;
    L = boxL;
    V = L * L * L;
    bjerrum = lB;
    rc = 0.5 * L;
    if (mode == EWALD) {
      alpha = 5.6 / L;
      kmax = 6;
      kvecs.clear();
      const double twopiL = 2 * PI / L;
      for (int nx = 0; nx <= kmax; ++nx)
        for (int ny = -kmax; ny <= kmax; ++ny)
          for (int nz = -kmax; nz <= kmax; ++nz) {
            if (nx == 0 && ny == 0 && nz == 0) continue;
            if (nx == 0 && (ny < 0 || (ny == 0 && nz < 0))) continue;  // half space
            KVec k;
            k.nx = nx; k.ny = ny; k.nz = nz;
            k.kx = twopiL * nx; k.ky = twopiL * ny; k.kz = twopiL * nz;
            double k2 = k.kx * k.kx + k.ky * k.ky + k.kz * k.kz;
            k.w = std::exp(-k2 / (4 * alpha * alpha)) / k2;
            kvecs.push_back(k);
          }
    } else {
      alpha = 0.2;
      double erc = std::erfc(alpha * rc);
      dsf_shiftU = erc / rc;
      dsf_shiftF = erc / (rc * rc) + 2 * alpha / std::sqrt(PI) * std::exp(-alpha * alpha * rc * rc) / rc;
      dsf_self = -(erc / (2 * rc) + alpha / std::sqrt(PI));
    }
  }

  inline void minimg(Vec3& r) const {
    r.x -= L * std::nearbyint(r.x / L);
    r.y -= L * std::nearbyint(r.y / L);
    r.z -= L * std::nearbyint(r.z / L);
  }

  // energy and forces (F resized to r.size()); positions may be unwrapped
  double compute(const std::vector<Vec3>& r, const std::vector<int>& q, std::vector<Vec3>& F) const {
    const int n = (int)r.size();
    F.assign(n, Vec3{});
    double U = 0;
    if (mode == EWALD) {
      const double a2 = alpha * alpha, twoa_sqrtpi = 2 * alpha / std::sqrt(PI);
      // real space (minimum image, r < rc)
      for (int i = 0; i < n; ++i)
        for (int j = i + 1; j < n; ++j) {
          Vec3 d = r[i] - r[j];
          minimg(d);
          double r2 = dot(d, d);
          if (r2 >= rc * rc) continue;
          double rr = std::sqrt(r2), inv = 1.0 / rr;
          double qq = q[i] * q[j];
          double erc = std::erfc(alpha * rr);
          U += qq * erc * inv;
          double fmag = qq * (erc * inv * inv + twoa_sqrtpi * std::exp(-a2 * r2) * inv) * inv;  // /r for unit vector
          F[i] += d * fmag;
          F[j] -= d * fmag;
        }
      // reciprocal space
      using cplx = std::complex<double>;
      const int nk = 2 * kmax + 1;
      std::vector<cplx> ex((size_t)n * nk), ey((size_t)n * nk), ez((size_t)n * nk);
      const double twopiL = 2 * PI / L;
      for (int i = 0; i < n; ++i) {
        cplx bx = std::polar(1.0, twopiL * r[i].x), by = std::polar(1.0, twopiL * r[i].y),
             bz = std::polar(1.0, twopiL * r[i].z);
        cplx px = 1, py = 1, pz = 1;
        for (int m = 0; m <= kmax; ++m) {
          ex[(size_t)i * nk + kmax + m] = px; ex[(size_t)i * nk + kmax - m] = std::conj(px);
          ey[(size_t)i * nk + kmax + m] = py; ey[(size_t)i * nk + kmax - m] = std::conj(py);
          ez[(size_t)i * nk + kmax + m] = pz; ez[(size_t)i * nk + kmax - m] = std::conj(pz);
          px *= bx; py *= by; pz *= bz;
        }
      }
      std::vector<cplx> ph(n);
      double Urec = 0;
      const double pref_f = 8 * PI / V;
      for (const KVec& k : kvecs) {
        cplx S = 0;
        for (int i = 0; i < n; ++i) {
          cplx p = ex[(size_t)i * nk + kmax + k.nx] * ey[(size_t)i * nk + kmax + k.ny] *
                   ez[(size_t)i * nk + kmax + k.nz];
          ph[i] = p;
          S += (double)q[i] * p;
        }
        Urec += k.w * std::norm(S);
        for (int i = 0; i < n; ++i) {
          // Im[exp(-i k.r_i) S]
          double im = (std::conj(ph[i]) * S).imag();
          double c = -pref_f * k.w * q[i] * im;
          F[i].x += c * k.kx; F[i].y += c * k.ky; F[i].z += c * k.kz;
        }
      }
      U += (4 * PI / V) * Urec;
      double sq2 = 0;
      for (int i = 0; i < n; ++i) sq2 += (double)q[i] * q[i];
      U += -alpha / std::sqrt(PI) * sq2;
    } else {
      const double a2 = alpha * alpha, twoa_sqrtpi = 2 * alpha / std::sqrt(PI);
      for (int i = 0; i < n; ++i)
        for (int j = i + 1; j < n; ++j) {
          Vec3 d = r[i] - r[j];
          minimg(d);
          double r2 = dot(d, d);
          if (r2 >= rc * rc) continue;
          double rr = std::sqrt(r2), inv = 1.0 / rr;
          double qq = q[i] * q[j];
          double erc = std::erfc(alpha * rr);
          U += qq * (erc * inv - dsf_shiftU + dsf_shiftF * (rr - rc));
          double fmag = qq * (erc * inv * inv + twoa_sqrtpi * std::exp(-a2 * r2) * inv - dsf_shiftF) * inv;
          F[i] += d * fmag;
          F[j] -= d * fmag;
        }
      double sq2 = 0;
      for (int i = 0; i < n; ++i) sq2 += (double)q[i] * q[i];
      U += dsf_self * sq2;
    }
    for (auto& f : F) f = f * bjerrum;
    return bjerrum * U;
  }
};

// ---------------------------------------------------------------------------
// Collision Verlet helpers
// ---------------------------------------------------------------------------
// Potential splitting of phi_c(q) = lB q_i q_j / q (HLL eqs. 19-25) with the Hermite cubic
//   P(q)  = phi_c(q2) + phi_c'(q2) [ (q-q1)^3 - (q2-q1)^3 ] / (3 (q2-q1)^2),
//   P'(q) = phi_c'(q2) (q-q1)^2 / (q2-q1)^2          (P(q2)=phi_c(q2), P'(q2)=phi_c'(q2), P'(q1)=P''(q1)=0)
//   phi_2 = P(q1) (q<q1) | P(q) (q1<=q<q2) | phi_c(q) (q>=q2),   phi_1 = phi_c - phi_2.
// qq = q_i q_j = +-1.  The force on i for the pair vector r = r_i - r_j is  F_1,i = f1_over_r(q) * r.
struct Split {
  double lB = 5.0, q1 = 1.0, q2 = 1.5;
  double phic(double q, double qq) const { return lB * qq / q; }
  double dphic(double q, double qq) const { return -lB * qq / (q * q); }
  double P(double q, double qq) const {
    const double d = q2 - q1, u = q - q1;
    return phic(q2, qq) + dphic(q2, qq) * (u * u * u - d * d * d) / (3 * d * d);
  }
  double dP(double q, double qq) const { const double d = q2 - q1, u = q - q1; return dphic(q2, qq) * u * u / (d * d); }
  double phi2(double q, double qq) const { return q < q1 ? P(q1, qq) : (q < q2 ? P(q, qq) : phic(q, qq)); }
  double phi1(double q, double qq) const { return q >= q2 ? 0.0 : phic(q, qq) - phi2(q, qq); }
  double dphi1(double q, double qq) const { return q >= q2 ? 0.0 : dphic(q, qq) - (q < q1 ? 0.0 : dP(q, qq)); }
  double f1_over_r(double q, double qq) const { return -dphi1(q, qq) / q; }
};

// Relative motion of a pair along quadratic paths: d(t) = dq + dv t + 0.5 da t^2.
struct QuadPath {
  Vec3 dq, dv, da;
  inline double f(double t) const { Vec3 p = dq + (dv + da * (0.5 * t)) * t; return dot(p, p) - SIGMA2; }
  inline double df(double t) const { Vec3 p = dq + (dv + da * (0.5 * t)) * t; return 2 * dot(p, dv + da * t); }
  inline double d2f(double t) const {
    Vec3 p = dq + (dv + da * (0.5 * t)) * t, w = dv + da * t;
    return 2 * (dot(w, w) + dot(p, da));
  }
};

// Root of a monotone function on [lo, hi] with f(lo) = flo, f(hi) = fhi of opposite sign (fhi may be 0):
// safeguarded Newton (rtsafe) with bisection fallback.  Returns the end of the final bracket on the
// flo side (for a collision entry root, flo > 0, this is the point just outside contact).
template <class F, class DF>
double bracket_root(F f, DF df, double lo, double hi, double flo, double fhi) {
  if (fhi == 0) return hi;
  double x = hi - fhi * (hi - lo) / (fhi - flo);
  if (!(x > lo && x < hi)) x = 0.5 * (lo + hi);
  double dxold = hi - lo, dx = dxold;
  for (int it = 0; it < 200; ++it) {
    double fx = f(x);
    if (fx == 0) return x;
    if ((fx > 0) == (flo > 0)) { lo = x; flo = fx; } else { hi = x; fhi = fx; }
    if (hi - lo <= 4 * std::numeric_limits<double>::epsilon() * std::fabs(hi) + 1e-300) break;
    double d = df(x);
    double xn;
    if (d == 0 || std::fabs(2 * fx) > std::fabs(dxold * d)) {
      xn = 0.5 * (lo + hi);
    } else {
      xn = x - fx / d;
      if (!(xn > lo && xn < hi)) xn = 0.5 * (lo + hi);
    }
    dxold = dx;
    dx = std::fabs(xn - x);
    x = xn;
  }
  return lo;
}

// Smallest tau in [0, tau_max] at which the pair distance along the quadratic relative path
// dq + dv tau + 0.5 da tau^2 crosses SIGMA from above (HLL eq. 29; da != 0 assumed).  The quartic
// f(tau) = |d(tau)|^2 - sigma^2 is split into monotone pieces at its critical points (roots of the
// cubic f', bracketed between the roots of the quadratic f''); each decreasing piece that brackets
// a sign change is solved by safeguarded Newton.  A pair inside the contact band
// |dq|^2 <= sigma^2 (1 + 1e-12) collides at tau = 0 if it approaches; if it separates, the next
// crossing from above (re-entry under attraction) is returned.  A decreasing piece that starts at
// f <= 0 (the pair never left contact: turning point inside the band) collides at its start.
bool quartic_collision_time(const Vec3& dq, const Vec3& dv, const Vec3& da, double tau_max, double& tau) {
  const double r2 = dot(dq, dq), b = dot(dq, dv);
  const double a2 = dot(da, da), v2 = dot(dv, dv), ra = dot(dq, da);
  if (r2 <= SIGMA2 * (1 + 1e-12)) {
    if (b < 0 || (b == 0 && v2 + ra < 0)) { tau = 0; return true; }
  } else {
    // triangle inequality: |d(tau)| >= |dq| - |dv| tau - 0.5 |da| tau^2
    const double reach = std::sqrt(v2) * tau_max + 0.5 * std::sqrt(a2) * tau_max * tau_max;
    if (std::sqrt(r2) - SIGMA > reach) return false;
  }
  if (!(tau_max > 0)) return false;
  const QuadPath P{dq, dv, da};
  auto f = [&](double t) { return P.f(t); };
  auto g = [&](double t) { return P.df(t); };
  auto dg = [&](double t) { return P.d2f(t); };
  // breakpoints of f' (roots of f'' = 12 c4 t^2 + 6 c3 t + 2 c2, c4 = |da|^2/4, c3 = dv.da, c2 = v2 + dq.da)
  double pts[6];
  int np = 0;
  pts[np++] = 0;
  {
    const double A = 3 * a2, B = 6 * dot(dv, da), C = 2 * (v2 + ra);
    const double disc = B * B - 4 * A * C;
    if (disc > 0 && A > 0) {
      const double sq = std::sqrt(disc);
      const double qq = -0.5 * (B + std::copysign(sq, B));
      double s1 = qq / A, s2 = (qq != 0) ? C / qq : s1;
      if (s1 > s2) std::swap(s1, s2);
      if (s1 > 0 && s1 < tau_max) pts[np++] = s1;
      if (s2 > 0 && s2 < tau_max && s2 != s1) pts[np++] = s2;
    }
  }
  pts[np++] = tau_max;
  // critical points of f in (0, tau_max)
  double crit[4];
  int ncrit = 0;
  {
    double ga = g(pts[0]);
    for (int k = 0; k + 1 < np; ++k) {
      const double a = pts[k], bnd = pts[k + 1];
      const double gb = g(bnd);
      if ((ga < 0 && gb > 0) || (ga > 0 && gb < 0)) crit[ncrit++] = bracket_root(g, dg, a, bnd, ga, gb);
      else if (gb == 0 && bnd < tau_max) crit[ncrit++] = bnd;
      ga = gb;
    }
  }
  double fp[6];
  int nf = 0;
  fp[nf++] = 0;
  for (int k = 0; k < ncrit; ++k) fp[nf++] = crit[k];
  fp[nf++] = tau_max;
  double fa = r2 - SIGMA2;
  for (int k = 0; k + 1 < nf; ++k) {
    const double a = fp[k], bnd = fp[k + 1];
    const double fb = f(bnd);
    if (fa > fb) {  // decreasing piece
      if (a > 0 && fa <= 0) { tau = a; return true; }
      if (fa > 0 && fb <= 0) { tau = bracket_root(f, g, a, bnd, fa, fb); return true; }
    }
    fa = fb;
  }
  return false;
}

// First time t >= 0 at which the 1-d quadratic motion x0 + v t + 0.5 a t^2 reaches the boundary b
// moving outward (sign of the velocity there = out, +1 for an upper, -1 for a lower cell face).
// Returns +inf if it never does.  A particle already outside and moving outward crosses at t = 0.
double quadratic_face_time(double x0, double v, double a, double b, int out) {
  const double c = x0 - b;  // > 0: above the face
  const double inf = std::numeric_limits<double>::infinity();
  if ((out > 0 && c >= 0) || (out < 0 && c <= 0)) {
    const double vel = v;
    if ((out > 0 && (vel > 0 || (vel == 0 && a > 0))) || (out < 0 && (vel < 0 || (vel == 0 && a < 0)))) return 0;
  }
  double best = inf;
  auto consider = [&](double t) {
    if (!(t > 0) || t >= best) return;
    const double w = v + a * t;
    if ((out > 0 && w > 0) || (out < 0 && w < 0)) best = t;
  };
  if (a == 0) {
    if (v != 0) consider(-c / v);
    return best;
  }
  const double A = 0.5 * a, B = v, C = c;
  const double disc = B * B - 4 * A * C;
  if (disc < 0) return inf;
  const double sq = std::sqrt(disc);
  const double qq = -0.5 * (B + std::copysign(sq, B));
  consider(qq / A);
  if (qq != 0) consider(C / qq);
  return best;
}

// ---------------------------------------------------------------------------
// Event-driven hard-sphere simulation
// ---------------------------------------------------------------------------
struct Event {
  double t;
  int i, j;          // j >= 0: pair collision; j < 0: cell crossing of i, dir = -1 - j
  uint32_t ci, cj;   // collision-counter snapshots for lazy invalidation
};
struct EventLater {
  bool operator()(const Event& a, const Event& b) const { return a.t > b.t; }
};

struct Sim {
  int N = 0;
  double L = 1, eta = 0;
  int ncell = 1;
  double cellw = 1;
  std::vector<Vec3> pos, vel;
  std::vector<double> tlast;
  std::vector<uint32_t> cnt;
  std::vector<int> cx, cy, cz, nxt, prv, head;
  std::vector<int> charge;     // 0, +1, -1
  std::vector<int> ions;       // indices of charged spheres
  std::vector<Vec3> acc;       // force (m=1) at the start of the current Verlet step; zero for neutrals
  std::vector<Vec3> adot;      // backward-difference time derivative of acc (previous step); zero for neutrals
  double h_prev = 0;
  bool have_prev = false;
  std::vector<Event> heap;     // binary min-heap (std::push_heap / pop_heap)
  double tnow = 0;
  // collision-Verlet bookkeeping: between kick_start and kick_end the stored velocity is the
  // leapfrog drift velocity u; the physical velocity at time t_kick+tau is u + acc*(tau - h/2).
  double t_kick = 0, h_step = 0;
  bool in_step = false;
  bool phys_collision = true;  // reflect physical velocities at collisions (false = naive drift reflection)
  uint64_t n_phys_separating = 0, n_drift_fix = 0;
  // ---- collision Verlet (HLL 2000) state; see cv_* below
  bool cv = false;
  Split split;
  std::vector<Vec3> a1;        // F_1 (m = 1) of each particle during the current Verlet sub-step; zero unless in A
  std::vector<char> inA;       // a1 != 0: quadratic path x + v dt + a1 dt^2/2, velocity v + a1 dt
  std::vector<int> Alist;      // the accelerated set A
  double t_sub0 = 0;           // start time of the current Verlet sub-step
  double t_step_end = 0;       // end of the current outer step = window of the quartic collision predictions
  double cv_vmin = 2e-5;       // minimum normal separation speed after a processed collision of an accelerated pair
  uint64_t n_substeps = 0, n_cv_vmin = 0, n_quartic = 0, n_cv_grazing = 0;
  std::vector<Vec3> F1tmp, rion_tmp;
  std::vector<int> changed;
  std::vector<uint32_t> mark;
  uint32_t mark_gen = 0;
  // statistics
  uint64_t ncoll = 0, ncross = 0, nstale = 0, nevents = 0, nskipped_grazing = 0;
  double virial = 0;           // sum over collisions of sigma*|dp|  (m = 1)
  double max_contact_err = 0;  // max | |r| - sigma | at processed collisions
  std::mt19937_64 rng;
  size_t heap_limit = 1 << 20;

  // ---------------- setup ----------------
  void setup_box(int n, double L_) {
    N = n;
    L = L_;
    ncell = (int)std::floor(L / SIGMA);
    if (ncell < 3) {
      std::fprintf(stderr, "error: box too small for cell list (L=%g -> ncell=%d < 3)\n", L, ncell);
      std::exit(2);
    }
    cellw = L / ncell;
    pos.assign(N, Vec3{});
    vel.assign(N, Vec3{});
    tlast.assign(N, 0.0);
    cnt.assign(N, 0u);
    cx.assign(N, 0); cy.assign(N, 0); cz.assign(N, 0);
    nxt.assign(N, -1); prv.assign(N, -1);
    head.assign((size_t)ncell * ncell * ncell, -1);
    charge.assign(N, 0);
    acc.assign(N, Vec3{});
    adot.assign(N, Vec3{});
    have_prev = false;
    a1.assign(N, Vec3{});
    inA.assign(N, 0);
    Alist.clear();
    mark.assign(N, 0u);
    ions.clear();
    heap.clear();
    heap_limit = std::max<size_t>(100000, (size_t)64 * N);
    tnow = 0;
  }

  static int lattice_n(int N) { return (int)std::ceil(std::cbrt((double)N) - 1e-9); }

  // simple-cubic lattice with small jitter, MB velocities at kT=1
  void init_lattice(uint64_t seed) {
    rng.seed(seed);
    int n = lattice_n(N);
    double a = L / n;
    if (a <= SIGMA) {
      std::fprintf(stderr, "error: lattice spacing %g <= sigma; cannot place %d spheres in L=%g\n", a, N, L);
      std::exit(2);
    }
    std::vector<int> sites(n * n * n);
    for (int s = 0; s < n * n * n; ++s) sites[s] = s;
    std::shuffle(sites.begin(), sites.end(), rng);
    double jit = 0.45 * std::min(a - SIGMA, 0.5);
    std::uniform_real_distribution<double> uj(-jit, jit);
    for (int i = 0; i < N; ++i) {
      int s = sites[i];
      int ix = s / (n * n), iy = (s / n) % n, iz = s % n;
      pos[i] = {(ix + 0.5) * a + uj(rng), (iy + 0.5) * a + uj(rng), (iz + 0.5) * a + uj(rng)};
      wrap(pos[i]);
    }
    std::normal_distribution<double> gauss(0.0, 1.0);
    Vec3 P{};
    for (int i = 0; i < N; ++i) {
      vel[i] = {gauss(rng), gauss(rng), gauss(rng)};
      P += vel[i];
    }
    for (int i = 0; i < N; ++i) vel[i] -= P * (1.0 / N);
    rescale_velocities();
    // overlap check (brute force)
    double dmin2 = min_dist2_bruteforce();
    if (dmin2 < SIGMA2) {
      std::fprintf(stderr, "error: initial lattice has overlaps (dmin=%g)\n", std::sqrt(dmin2));
      std::exit(2);
    }
    std::fill(tlast.begin(), tlast.end(), tnow);
    assign_cells();
    rebuild_events();
  }

  void wrap(Vec3& p) const {
    p.x -= L * std::floor(p.x / L);
    p.y -= L * std::floor(p.y / L);
    p.z -= L * std::floor(p.z / L);
  }
  inline void minimg(Vec3& r) const {
    if (r.x > 0.5 * L) r.x -= L; else if (r.x < -0.5 * L) r.x += L;
    if (r.y > 0.5 * L) r.y -= L; else if (r.y < -0.5 * L) r.y += L;
    if (r.z > 0.5 * L) r.z -= L; else if (r.z < -0.5 * L) r.z += L;
  }
  inline void minimg_full(Vec3& r) const {
    r.x -= L * std::nearbyint(r.x / L);
    r.y -= L * std::nearbyint(r.y / L);
    r.z -= L * std::nearbyint(r.z / L);
  }

  double min_dist2_bruteforce() const {
    double dmin2 = std::numeric_limits<double>::infinity();
    for (int i = 0; i < N; ++i)
      for (int j = i + 1; j < N; ++j) {
        Vec3 d = pos_at(i, tnow) - pos_at(j, tnow);
        minimg_full(d);
        dmin2 = std::min(dmin2, dot(d, d));
      }
    return dmin2;
  }

  double kinetic() const {
    double ke = 0;
    for (int i = 0; i < N; ++i) ke += dot(vel[i], vel[i]);
    return 0.5 * ke;
  }
  Vec3 momentum() const {
    Vec3 P{};
    for (int i = 0; i < N; ++i) P += vel[i];
    return P;
  }
  double target_ke() const { return 1.5 * (N - 1) * KT; }
  void rescale_velocities() {
    double ke = kinetic();
    double s = std::sqrt(target_ke() / ke);
    for (int i = 0; i < N; ++i) vel[i] = vel[i] * s;
  }

  // ---------------- cells ----------------
  inline int cell_index(int a, int b, int c) const { return (a * ncell + b) * ncell + c; }
  inline int wrapc(int c) const { return c < 0 ? c + ncell : (c >= ncell ? c - ncell : c); }
  void cell_insert(int i) {
    int c = cell_index(cx[i], cy[i], cz[i]);
    nxt[i] = head[c];
    prv[i] = -1;
    if (head[c] >= 0) prv[head[c]] = i;
    head[c] = i;
  }
  void cell_remove(int i) {
    int c = cell_index(cx[i], cy[i], cz[i]);
    if (prv[i] >= 0) nxt[prv[i]] = nxt[i]; else head[c] = nxt[i];
    if (nxt[i] >= 0) prv[nxt[i]] = prv[i];
    nxt[i] = prv[i] = -1;
  }
  void assign_cells() {
    std::fill(head.begin(), head.end(), -1);
    for (int i = 0; i < N; ++i) {
      Vec3 p = pos[i];
      wrap(p);
      pos[i] = p;
      cx[i] = std::min(ncell - 1, (int)(p.x / cellw));
      cy[i] = std::min(ncell - 1, (int)(p.y / cellw));
      cz[i] = std::min(ncell - 1, (int)(p.z / cellw));
      cell_insert(i);
    }
  }

  // ---------------- kinematics ----------------
  // Lazy positions: ballistic for everybody except the accelerated set A of the collision-Verlet
  // sub-step (quadratic path with the constant sub-step acceleration a1; the stored velocity is the
  // velocity at tlast, i.e. at the sub-step start or at the last sync along the same parabola).
  inline Vec3 pos_at(int i, double t) const {
    const double dt = t - tlast[i];
    Vec3 p = pos[i] + vel[i] * dt;
    if (inA[i]) p += a1[i] * (0.5 * dt * dt);
    return p;
  }
  inline Vec3 vel_at(int i, double t) const { return inA[i] ? vel[i] + a1[i] * (t - tlast[i]) : vel[i]; }
  inline void sync(int i, double t) {
    pos[i] = pos_at(i, t);
    if (inA[i]) vel[i] += a1[i] * (t - tlast[i]);
    tlast[i] = t;
  }
  void sync_all(double t) { for (int i = 0; i < N; ++i) sync(i, t); }

  // ---------------- events ----------------
  inline void push_event(const Event& e) {
    heap.push_back(e);
    std::push_heap(heap.begin(), heap.end(), EventLater());
  }

  void schedule_crossing(int i) {
    double best = std::numeric_limits<double>::infinity();
    int bdir = -1;
    const int cc[3] = {cx[i], cy[i], cz[i]};
    if (inA[i]) {
      // quadratic path: per axis both faces, first outward crossing (velocity may reverse)
      for (int d = 0; d < 3; ++d) {
        const double x0 = comp(pos[i], d), v = comp(vel[i], d), a = comp(a1[i], d);
        const double thi = quadratic_face_time(x0, v, a, (cc[d] + 1) * cellw, +1);
        const double tlo = quadratic_face_time(x0, v, a, cc[d] * cellw, -1);
        if (thi < best) { best = thi; bdir = 2 * d + 1; }
        if (tlo < best) { best = tlo; bdir = 2 * d; }
      }
      if (bdir < 0) return;
      best += tlast[i];
      if (best < tnow) best = tnow;
      push_event(Event{best, i, -1 - bdir, cnt[i], 0u});
      return;
    }
    for (int d = 0; d < 3; ++d) {
      double v = comp(vel[i], d);
      if (v == 0) continue;
      double bnd = (v > 0) ? (cc[d] + 1) * cellw : cc[d] * cellw;
      double tc = tlast[i] + (bnd - comp(pos[i], d)) / v;
      if (tc < tnow) tc = tnow;
      if (tc < best) { best = tc; bdir = 2 * d + (v > 0 ? 1 : 0); }
    }
    if (bdir >= 0) push_event(Event{best, i, -1 - bdir, cnt[i], 0u});
  }

  // collision prediction for a pair with at least one accelerated member (collision Verlet):
  // quartic on the window (tnow, t_step_end] (all ions are re-kicked and re-predicted at the step end)
  inline bool predict_pair_cv(int i, int j, double& tc) {
    Vec3 r = pos_at(i, tnow) - pos_at(j, tnow);
    minimg(r);
    const Vec3 v = vel_at(i, tnow) - vel_at(j, tnow);
    const Vec3 da = a1[i] - a1[j];
    if (da.x == 0 && da.y == 0 && da.z == 0) {
      // identical accelerations: ordinary EDMD quadratic (exact for all times)
      double b = dot(r, v);
      if (b >= 0) return false;
      double v2 = dot(v, v);
      if (v2 <= 0) return false;
      double r2 = dot(r, r);
      if (r2 <= SIGMA2 * (1 + 1e-12)) { tc = tnow; return true; }
      double disc = b * b - v2 * (r2 - SIGMA2);
      if (disc < 0) return false;
      tc = tnow + (r2 - SIGMA2) / (-b + std::sqrt(disc));
      return true;
    }
    ++n_quartic;
    double tau;
    if (!quartic_collision_time(r, v, da, t_step_end - tnow, tau)) return false;
    tc = tnow + tau;
    return true;
  }

  // predicted collision of (i,j) given current velocities; positions evaluated at tnow
  inline bool predict_pair(int i, int j, double& tc) {
    if (inA[i] || inA[j]) return predict_pair_cv(i, j, tc);
    Vec3 r = pos_at(i, tnow) - pos_at(j, tnow);
    minimg(r);
    Vec3 v = vel[i] - vel[j];
    double b = dot(r, v);
    if (b >= 0) return false;
    double v2 = dot(v, v);
    if (v2 <= 0) return false;
    double r2 = dot(r, r);
    if (r2 <= SIGMA2 * (1 + 1e-12)) { tc = tnow; return true; }  // roundoff overlap, approaching
    double disc = b * b - v2 * (r2 - SIGMA2);
    if (disc < 0) return false;
    tc = tnow + (r2 - SIGMA2) / (-b + std::sqrt(disc));
    return true;
  }

  void predict_collisions(int i) {
    for (int dx = -1; dx <= 1; ++dx)
      for (int dy = -1; dy <= 1; ++dy)
        for (int dz = -1; dz <= 1; ++dz) {
          int c = cell_index(wrapc(cx[i] + dx), wrapc(cy[i] + dy), wrapc(cz[i] + dz));
          for (int j = head[c]; j >= 0; j = nxt[j]) {
            if (j == i) continue;
            double tc;
            if (predict_pair(i, j, tc)) push_event(Event{tc, i, j, cnt[i], cnt[j]});
          }
        }
  }

  void rebuild_events() {
    heap.clear();
    for (int i = 0; i < N; ++i) schedule_crossing(i);
    for (int i = 0; i < N; ++i)
      for (int dx = -1; dx <= 1; ++dx)
        for (int dy = -1; dy <= 1; ++dy)
          for (int dz = -1; dz <= 1; ++dz) {
            int c = cell_index(wrapc(cx[i] + dx), wrapc(cy[i] + dy), wrapc(cz[i] + dz));
            for (int j = head[c]; j >= 0; j = nxt[j]) {
              if (j <= i) continue;
              double tc;
              if (predict_pair(i, j, tc)) heap.push_back(Event{tc, i, j, cnt[i], cnt[j]});
            }
          }
    std::make_heap(heap.begin(), heap.end(), EventLater());
  }

  inline bool stale(const Event& e) const {
    if (e.ci != cnt[e.i]) return true;
    return e.j >= 0 && e.cj != cnt[e.j];
  }

  void compact_heap() {
    size_t k = 0;
    for (size_t a = 0; a < heap.size(); ++a)
      if (!stale(heap[a])) heap[k++] = heap[a];
    heap.resize(k);
    std::make_heap(heap.begin(), heap.end(), EventLater());
    heap_limit = std::max<size_t>(100000, std::max((size_t)64 * N, 2 * heap.size()));
  }

  // ---------------- collision Verlet: H_1 = T + V_hs + V_1 propagation (HLL Table 1) ----------------
  // F_1 of all ions at time t (positions evaluated on the lazy paths; the accelerated ones must be synced)
  void compute_F1(double t, std::vector<Vec3>& F1) {
    const size_t n = ions.size();
    rion_tmp.resize(n);
    for (size_t k = 0; k < n; ++k) rion_tmp[k] = pos_at(ions[k], t);
    F1.assign(n, Vec3{});
    const double q22 = split.q2 * split.q2;
    for (size_t a = 0; a < n; ++a)
      for (size_t b = a + 1; b < n; ++b) {
        Vec3 d = rion_tmp[a] - rion_tmp[b];
        minimg(d);
        const double r2 = dot(d, d);
        if (r2 >= q22) continue;
        const double qq = (double)(charge[ions[a]] * charge[ions[b]]);
        const Vec3 fd = d * split.f1_over_r(std::sqrt(r2), qq);
        F1[a] += fd;
        F1[b] -= fd;
      }
  }
  inline void mark_changed(int i) {
    if (mark[i] != mark_gen) { mark[i] = mark_gen; changed.push_back(i); }
  }
  // End of the Verlet sub-step that started at t_sub0, at the current time t = tnow (all events < t done):
  // advance the accelerated set along its parabolas (p_half + drift of Table 1), recompute F_1 at the new
  // positions and merge the second half kick of this sub-step with the first of the next one,
  //     v <- v + tau/2 (a_old + a_new)   written as   v_phys(t) + tau/2 (a_new - a_old),
  // for every ion whose F_1 is or was non-zero (entering / leaving A is included).  Collects the particles
  // whose velocity or acceleration changed in `changed`; the caller bumps counters and re-predicts.
  void cv_substep_boundary(double t) {
    const double tau = t - t_sub0;
    for (int i : Alist) sync(i, t);
    compute_F1(t, F1tmp);
    ++mark_gen;
    changed.clear();
    Alist.clear();
    for (size_t k = 0; k < ions.size(); ++k) {
      const int i = ions[k];
      const Vec3 an = F1tmp[k], ao = a1[i];
      const bool was = inA[i] != 0, now = (an.x != 0 || an.y != 0 || an.z != 0);
      if (now) Alist.push_back(i);
      if (!was && !now) continue;
      if (!was) sync(i, t);  // entering A: parabola starts here
      vel[i] += (an - ao) * (0.5 * tau);
      a1[i] = an;
      inA[i] = now ? 1 : 0;
      mark_changed(i);
    }
    t_sub0 = t;
    ++n_substeps;
  }
  void repredict_changed() {
    for (int i : changed) ++cnt[i];
    for (int i : changed) { schedule_crossing(i); predict_collisions(i); }
    changed.clear();
  }
  // switch the CV machinery on (after assign_charges): F_1 at the current positions defines A
  void cv_begin(double lB, double q1, double q2) {
    cv = true;
    split.lB = lB; split.q1 = q1; split.q2 = q2;
    sync_all(tnow);
    t_sub0 = tnow;
    t_step_end = tnow;
    cv_substep_boundary(tnow);
    repredict_changed();
  }
  // outer half kick with F_2 = F_total - F_1 at the current (synchronized) positions and start of a step of
  // length h (F_total indexed like `ions`; F_1 = a1 of the ions, consistent with the same positions)
  void cv_kick_start(const std::vector<Vec3>& Ftot, double h) {
    for (size_t k = 0; k < ions.size(); ++k) {
      const int i = ions[k];
      sync(i, tnow);
      vel[i] += (Ftot[k] - a1[i]) * (0.5 * h);
      ++cnt[i];
    }
    t_sub0 = tnow;
    t_step_end = tnow + h;
    in_step = true;
    for (int i : ions) { schedule_crossing(i); predict_collisions(i); }
  }
  // final Verlet sub-step of the step (ends at the current tnow = step end) -- call after advance_to(step end)
  void cv_finish_step() {
    cv_substep_boundary(tnow);
    changed.clear();  // the second half kick re-predicts every ion anyway
  }
  // second outer half kick with F_2 at the end-of-step positions; velocities are physical afterwards
  void cv_kick_end(const std::vector<Vec3>& Ftot, double h) {
    for (size_t k = 0; k < ions.size(); ++k) {
      const int i = ions[k];
      sync(i, tnow);
      vel[i] += (Ftot[k] - a1[i]) * (0.5 * h);
      ++cnt[i];
    }
    in_step = false;
    t_step_end = tnow;  // no window until the next cv_kick_start (which re-predicts all ions)
    for (int i : ions) { schedule_crossing(i); predict_collisions(i); }
  }

  // one hard collision inside the collision-Verlet propagation: it ends the current Verlet sub-step for
  // the whole system (Table 1), then the pair is reflected with the updated (physical) velocities
  void process_collision_cv(const Event& e) {
    const int i = e.i, j = e.j;
    sync(i, e.t);
    sync(j, e.t);
    cv_substep_boundary(e.t);
    Vec3 r = pos[i] - pos[j];
    minimg(r);
    const double rn = std::sqrt(dot(r, r));
    max_contact_err = std::max(max_contact_err, std::fabs(rn - SIGMA));
    const Vec3 n = r * (1.0 / rn);
    double vn = dot(vel[i] - vel[j], n);
    const bool accelerated = inA[i] || inA[j];
    if (vn < 0) {
      const Vec3 dv = n * vn;
      vel[i] -= dv;
      vel[j] += dv;
      virial += SIGMA * (-vn);
      ++ncoll;
      vn = -vn;
      mark_changed(i);
      mark_changed(j);
    } else if (!accelerated) {
      ++nskipped_grazing;  // ballistic pair already separating (grazing roundoff): no impulse
    } else {
      ++n_cv_grazing;
    }
    if (accelerated && vn < cv_vmin) {
      // Under an inward relative acceleration a pair leaving contact at normal speed vn returns after
      // 2 vn / a_n; below cv_vmin the excursion (sigma vn^2 / a_n) is at the roundoff level of the quartic
      // and the bounce frequency diverges.  Enforce a minimum separation speed (momentum conserving;
      // energy input <= cv_vmin^2 / 4 per event).
      const double sft = 0.5 * (cv_vmin - vn);
      vel[i] += n * sft;
      vel[j] -= n * sft;
      ++n_cv_vmin;
      mark_changed(i);
      mark_changed(j);
    }
    repredict_changed();
  }

  void process_collision(const Event& e) {
    if (cv) { process_collision_cv(e); return; }
    const int i = e.i, j = e.j;
    sync(i, e.t);
    sync(j, e.t);
    Vec3 r = pos[i] - pos[j];
    minimg(r);
    double rn = std::sqrt(dot(r, r));
    max_contact_err = std::max(max_contact_err, std::fabs(rn - SIGMA));
    Vec3 n = r * (1.0 / rn);
    const bool ion_step = in_step && phys_collision && (charge[i] != 0 || charge[j] != 0);
    if (!ion_step) {
      // plain elastic reflection of the stored velocities (always exact for neutral pairs)
      Vec3 v = vel[i] - vel[j];
      double vn = dot(v, n);
      if (vn >= 0) { ++nskipped_grazing; return; }  // separating already (grazing roundoff): no impulse
      Vec3 dv = n * vn;
      vel[i] -= dv;
      vel[j] += dv;
      virial += SIGMA * (-vn);
      ++ncoll;
    } else {
      // Collision Verlet: reflect the *physical* velocities v = u + c with
      // c = a (tau1 - h/2) + adot tau1^2/2  (leapfrog drift velocity u -> velocity at the collision
      // time, force extrapolated linearly within the step), then return to drift velocities with
      // w = v+ - c (this also gives the correct end-of-step velocity after the second half kick).
      // Pair momentum is conserved exactly; the drift position at the step end is unchanged to
      // O(dt^2) (the pre/post-collision displacements a*tau1*tau2/2 cancel).
      const double tau1 = e.t - t_kick, sh = tau1 - 0.5 * h_step, q2 = 0.5 * tau1 * tau1;
      const Vec3 ci = acc[i] * sh + adot[i] * q2, cj = acc[j] * sh + adot[j] * q2;
      Vec3 vi = vel[i] + ci, vj = vel[j] + cj;
      double vn = dot(vi - vj, n);
      if (vn < 0) {
        Vec3 dv = n * vn;
        vi -= dv;
        vj += dv;
        virial += SIGMA * (-vn);
        ++ncoll;
        vel[i] = vi - ci;
        vel[j] = vj - cj;
      } else {
        ++n_phys_separating;  // physically already separating: no impulse
      }
      // the EDMD moves the spheres with the drift velocities: make sure those separate (no overlap)
      double wn = dot(vel[i] - vel[j], n);
      const double eps = 1e-9;
      if (wn < eps) {
        double sft = 0.5 * (wn - eps);
        vel[i] -= n * sft;
        vel[j] += n * sft;
        ++n_drift_fix;
      }
    }
    ++cnt[i]; ++cnt[j];
    schedule_crossing(i);
    schedule_crossing(j);
    predict_collisions(i);
    predict_collisions(j);
  }

  void process_crossing(const Event& e) {
    const int i = e.i;
    const int dir = -1 - e.j, d = dir >> 1, sgn = (dir & 1) ? 1 : -1;
    sync(i, e.t);
    cell_remove(i);
    int* cc = (d == 0) ? &cx[i] : (d == 1 ? &cy[i] : &cz[i]);
    int c = *cc + sgn;
    if (c >= ncell) { c = 0; comp(pos[i], d) -= L; }
    else if (c < 0) { c = ncell - 1; comp(pos[i], d) += L; }
    *cc = c;
    cell_insert(i);
    ++ncross;
    schedule_crossing(i);
    predict_collisions(i);
  }

  // process all events with t < t_target, then set tnow = t_target (t_target >= tnow required:
  // running the clock backwards would leave tlast/cells/heap at the later time)
  void advance_to(double t_target) {
    if (t_target < tnow) {
      if (tnow - t_target > 1e-9) {
        std::fprintf(stderr, "internal error: advance_to(%.12g) called with tnow=%.12g (backwards)\n", t_target, tnow);
        std::exit(3);
      }
      return;  // roundoff-level: keep tnow, do not move backwards
    }
    while (!heap.empty() && heap.front().t < t_target) {
      Event e = heap.front();
      std::pop_heap(heap.begin(), heap.end(), EventLater());
      heap.pop_back();
      if (stale(e)) { ++nstale; continue; }
      if (e.t < tnow) e.t = tnow;  // guard (should not happen)
      tnow = e.t;
      ++nevents;
      if (e.j >= 0) process_collision(e); else process_crossing(e);
      if (heap.size() > heap_limit) compact_heap();
    }
    tnow = t_target;
  }

  // ---------------- charges / kicks ----------------
  void assign_charges(int nplus, int nminus) {
    std::vector<int> idx(N);
    for (int i = 0; i < N; ++i) idx[i] = i;
    std::shuffle(idx.begin(), idx.end(), rng);
    ions.clear();
    for (int k = 0; k < nplus; ++k) { charge[idx[k]] = +1; ions.push_back(idx[k]); }
    for (int k = nplus; k < nplus + nminus; ++k) { charge[idx[k]] = -1; ions.push_back(idx[k]); }
  }

  void ion_positions(double t, std::vector<Vec3>& r) const {
    r.resize(ions.size());
    for (size_t k = 0; k < ions.size(); ++k) r[k] = pos_at(ions[k], t);
  }
  void ion_charges(std::vector<int>& q) const {
    q.resize(ions.size());
    for (size_t k = 0; k < ions.size(); ++k) q[k] = charge[ions[k]];
  }

  // v_ion += s*F ; positions synced to tnow; events re-predicted for the kicked ions only
  void apply_kick(const std::vector<Vec3>& F, double s, bool store_acc) {
    for (size_t k = 0; k < ions.size(); ++k) {
      int i = ions[k];
      sync(i, tnow);
      vel[i] += F[k] * s;
      if (store_acc) {
        adot[i] = have_prev ? (F[k] - acc[i]) * (1.0 / h_prev) : Vec3{};
        acc[i] = F[k];
      }
      ++cnt[i];
    }
    for (int i : ions) {
      schedule_crossing(i);
      predict_collisions(i);
    }
  }
  // first half kick of a step of length h (F = forces at the current, synchronized positions)
  void kick_start(const std::vector<Vec3>& F, double h) {
    apply_kick(F, 0.5 * h, true);
    t_kick = tnow;
    h_step = h;
    h_prev = h;
    have_prev = true;
    in_step = true;
  }
  // second half kick (F = forces at the end-of-step positions); velocities are physical afterwards
  void kick_end(const std::vector<Vec3>& F, double h) {
    apply_kick(F, 0.5 * h, false);
    in_step = false;
  }

  // ---------------- diagnostics ----------------
  double min_dist_celllist(double t) const {
    double dmin2 = std::numeric_limits<double>::infinity();
    for (int i = 0; i < N; ++i) {
      Vec3 pi = pos_at(i, t);
      for (int dx = -1; dx <= 1; ++dx)
        for (int dy = -1; dy <= 1; ++dy)
          for (int dz = -1; dz <= 1; ++dz) {
            int c = cell_index(wrapc(cx[i] + dx), wrapc(cy[i] + dy), wrapc(cz[i] + dz));
            for (int j = head[c]; j >= 0; j = nxt[j]) {
              if (j <= i) continue;
              Vec3 d = pi - pos_at(j, t);
              minimg(d);
              dmin2 = std::min(dmin2, dot(d, d));
            }
          }
    }
    return std::sqrt(dmin2);
  }
  int count_pm_pairs(double t, double rcut) const {
    int n = 0;
    for (size_t a = 0; a < ions.size(); ++a)
      for (size_t b = a + 1; b < ions.size(); ++b) {
        int i = ions[a], j = ions[b];
        if (charge[i] * charge[j] >= 0) continue;
        Vec3 d = pos_at(i, t) - pos_at(j, t);
        minimg_full(d);
        if (dot(d, d) < rcut * rcut) ++n;
      }
    return n;
  }
};

// ---------------------------------------------------------------------------
// Options
// ---------------------------------------------------------------------------
struct Options {
  std::string name = "run";
  int N = 1000;
  double eta = 0.35;
  int nplus = 0, nminus = 0;
  double bjerrum = 5.0;
  std::string coulomb = "ewald";
  std::string integrator = "cv";   // cv | kdk-phys | naive
  double cv_q1 = 1.0, cv_q2 = 1.5;  // potential-splitting radii (sigma)
  double dt = 0.005;
  double t_equil_neutral = 5, t_equil_charged = 15, t_run = 18, dt_frame = 0.025;
  double thermostat_interval = 0.1;
  uint64_t seed = 1;
  std::string xyz, diag, summary;
  bool selftest = false, quiet = false;
};

void usage() {
  std::printf(
      "hsdmd %s -- event-driven hard spheres (+ RPM charges via collision Verlet)\n"
      "  --name S            run name (metadata only)\n"
      "  --N I               number of spheres (1000)\n"
      "  --eta F             packing fraction (0.35)\n"
      "  --nplus I --nminus I   number of +e / -e spheres (0, 0)\n"
      "  --bjerrum F         Bjerrum length l_B/sigma (5.0)\n"
      "  --coulomb ewald|dsf (ewald)\n"
      "  --integrator cv|kdk-phys|naive   (cv) cv = collision Verlet of Houndonougbo, Laird & Leimkuhler\n"
      "                      2000 (potential splitting + Verlet sub-steps between collisions, O(dt^2));\n"
      "                      kdk-phys = kick-drift-kick with the physical-velocity collision rule;\n"
      "                      naive = kick-drift-kick with drift-velocity reflection (HLL eq. 9, O(dt))\n"
      "  --cv-q1 F --cv-q2 F   splitting radii in sigma (1.0, 1.5): phi_2 is flat for r < q1, phi_1 = 0 for r >= q2\n"
      "  --collision-rule phys|naive   deprecated alias of --integrator kdk-phys | naive\n"
      "  --dt F              Verlet step for charged runs (0.005); rounded so dt_frame/dt is integer\n"
      "  --t-equil-neutral F --t-equil-charged F --t-run F   (5, 15, 18); the +/- pairing\n"
      "                      relaxation time is 4-6 tau, so t-equil-charged >= 10 is recommended\n"
      "  --dt-frame F        frame spacing (0.025)\n"
      "  --seed I            RNG seed (1)\n"
      "  --xyz PATH --diag PATH --summary PATH   output files\n"
      "  --selftest          run unit checks and exit (0 = pass)\n"
      "  --quiet             no per-tau progress on stderr\n",
      HSDMD_VERSION);
}

Options parse(int argc, char** argv) {
  Options o;
  auto need = [&](int& k) -> const char* {
    if (k + 1 >= argc) { std::fprintf(stderr, "missing value for %s\n", argv[k]); std::exit(2); }
    return argv[++k];
  };
  for (int k = 1; k < argc; ++k) {
    std::string a = argv[k];
    if (a == "--help" || a == "-h") { usage(); std::exit(0); }
    else if (a == "--selftest") o.selftest = true;
    else if (a == "--quiet") o.quiet = true;
    else if (a == "--name") o.name = need(k);
    else if (a == "--N") o.N = std::atoi(need(k));
    else if (a == "--eta") o.eta = std::atof(need(k));
    else if (a == "--nplus") o.nplus = std::atoi(need(k));
    else if (a == "--nminus") o.nminus = std::atoi(need(k));
    else if (a == "--bjerrum") o.bjerrum = std::atof(need(k));
    else if (a == "--coulomb") o.coulomb = need(k);
    else if (a == "--integrator") o.integrator = need(k);
    else if (a == "--cv-q1") o.cv_q1 = std::atof(need(k));
    else if (a == "--cv-q2") o.cv_q2 = std::atof(need(k));
    else if (a == "--collision-rule") {
      std::string r = need(k);
      if (r != "phys" && r != "naive") { std::fprintf(stderr, "--collision-rule must be phys or naive\n"); std::exit(2); }
      o.integrator = (r == "phys") ? "kdk-phys" : "naive";
      std::fprintf(stderr, "note: --collision-rule %s is deprecated; using --integrator %s\n", r.c_str(), o.integrator.c_str());
    }
    else if (a == "--dt") o.dt = std::atof(need(k));
    else if (a == "--t-equil-neutral") o.t_equil_neutral = std::atof(need(k));
    else if (a == "--t-equil-charged") o.t_equil_charged = std::atof(need(k));
    else if (a == "--t-run") o.t_run = std::atof(need(k));
    else if (a == "--dt-frame") o.dt_frame = std::atof(need(k));
    else if (a == "--seed") o.seed = std::strtoull(need(k), nullptr, 10);
    else if (a == "--xyz") o.xyz = need(k);
    else if (a == "--diag") o.diag = need(k);
    else if (a == "--summary") o.summary = need(k);
    else { std::fprintf(stderr, "unknown option %s\n", a.c_str()); usage(); std::exit(2); }
  }
  if (o.coulomb != "ewald" && o.coulomb != "dsf") { std::fprintf(stderr, "--coulomb must be ewald or dsf\n"); std::exit(2); }
  if (o.integrator != "cv" && o.integrator != "kdk-phys" && o.integrator != "naive") {
    std::fprintf(stderr, "--integrator must be cv, kdk-phys or naive\n");
    std::exit(2);
  }
  auto die = [](const char* msg) { std::fprintf(stderr, "error: %s\n", msg); std::exit(2); };
  if (!(o.cv_q1 >= SIGMA) || !(o.cv_q2 > o.cv_q1) || !std::isfinite(o.cv_q2)) die("need sigma <= --cv-q1 < --cv-q2");
  if (o.N < 1) die("--N must be >= 1");
  if (o.nplus < 0 || o.nminus < 0) die("--nplus/--nminus must be >= 0");
  if (o.nplus + o.nminus > o.N) die("nplus+nminus > N");
  if (!(o.eta > 0 && o.eta < 0.74)) die("--eta must be in (0, 0.74)");
  if (!(o.dt > 0) || !(o.dt_frame > 0)) die("--dt and --dt-frame must be > 0");
  if (!(o.t_equil_neutral >= 0) || !(o.t_equil_charged >= 0) || !(o.t_run >= 0))
    die("--t-equil-neutral, --t-equil-charged and --t-run must be >= 0");
  if (!(o.bjerrum >= 0)) die("--bjerrum must be >= 0");
  if (!(o.thermostat_interval > 0)) die("thermostat interval must be > 0");
  return o;
}

double carnahan_starling_Z(double eta) {
  return (1 + eta + eta * eta - eta * eta * eta) / std::pow(1 - eta, 3);
}
double enskog_rate(double eta) {  // collisions per particle per tau, kT=m=sigma=1
  double g = (1 - 0.5 * eta) / std::pow(1 - eta, 3);
  return 24.0 * eta * g / std::sqrt(PI);
}

// ---------------------------------------------------------------------------
// Self tests
// ---------------------------------------------------------------------------
int selftest() {
  int fails = 0;
  auto check = [&](bool ok, const char* what, double val) {
    std::printf("  [%s] %-58s %.3e\n", ok ? "PASS" : "FAIL", what, val);
    if (!ok) ++fails;
  };
  std::printf("hsdmd --selftest\n");

  // 1) finite-difference force checks, Ewald and DSF, 40 ions in the N=1000/eta=0.35 box
  {
    const double L = std::cbrt(1000 * PI / (6 * 0.35));
    std::mt19937_64 rng(12345);
    std::uniform_real_distribution<double> u(0, L);
    std::vector<Vec3> r;
    std::vector<int> q;
    while ((int)r.size() < 40) {
      Vec3 p{u(rng), u(rng), u(rng)};
      bool ok = true;
      for (auto& o : r) {
        Vec3 d = p - o;
        d.x -= L * std::nearbyint(d.x / L); d.y -= L * std::nearbyint(d.y / L); d.z -= L * std::nearbyint(d.z / L);
        if (dot(d, d) < 1.0) { ok = false; break; }
      }
      if (ok) { r.push_back(p); q.push_back((int)r.size() <= 20 ? +1 : -1); }
    }
    for (int mode = 0; mode < 2; ++mode) {
      Coulomb cb;
      cb.setup(mode == 0 ? Coulomb::EWALD : Coulomb::DSF, L, 5.0);
      std::vector<Vec3> F, Fd;
      double U0 = cb.compute(r, q, F);
      double fmax = 0, errmax = 0;
      for (auto& f : F) fmax = std::max(fmax, std::max(std::fabs(f.x), std::max(std::fabs(f.y), std::fabs(f.z))));
      const double h = 1e-5;
      for (size_t i = 0; i < r.size(); ++i)
        for (int d = 0; d < 3; ++d) {
          std::vector<Vec3> rp = r, rm = r;
          comp(rp[i], d) += h;
          comp(rm[i], d) -= h;
          double Up = cb.compute(rp, q, Fd), Um = cb.compute(rm, q, Fd);
          double fnum = -(Up - Um) / (2 * h);
          errmax = std::max(errmax, std::fabs(fnum - comp(F[i], d)));
        }
      Vec3 Ftot{};
      for (auto& f : F) Ftot += f;
      char buf[128];
      std::snprintf(buf, sizeof buf, "%s: max|F_num-F|/max|F| (U=%.6f)", mode == 0 ? "ewald" : "dsf", U0);
      check(errmax / fmax < 1e-6, buf, errmax / fmax);
      std::snprintf(buf, sizeof buf, "%s: |sum F| / max|F| (momentum conservation)", mode == 0 ? "ewald" : "dsf");
      double ft = std::sqrt(dot(Ftot, Ftot)) / fmax;
      check(ft < 1e-10, buf, ft);
    }
  }
  // 2) Ewald Madelung constant: 2x2x2 NaCl supercell, L=4, nearest-neighbour distance 1
  {
    std::vector<Vec3> r;
    std::vector<int> q;
    for (int x = 0; x < 4; ++x)
      for (int y = 0; y < 4; ++y)
        for (int z = 0; z < 4; ++z) { r.push_back({(double)x, (double)y, (double)z}); q.push_back(((x + y + z) & 1) ? -1 : 1); }
    Coulomb cb;
    cb.setup(Coulomb::EWALD, 4.0, 1.0);
    std::vector<Vec3> F;
    double U = cb.compute(r, q, F);
    double madelung = -2.0 * U / (double)r.size();  // per ion pair, d = 1
    check(std::fabs(madelung - 1.747564594633) < 2e-3, "ewald: NaCl Madelung constant vs 1.7475646", madelung);
  }
  // 3) two-sphere head-on collision (plus one across the periodic boundary)
  for (int variant = 0; variant < 2; ++variant) {
    Sim s;
    s.setup_box(2, 6.0);
    s.rng.seed(1);
    if (variant == 0) { s.pos[0] = {2.0, 3.0, 3.0}; s.pos[1] = {4.0, 3.0, 3.0}; s.vel[0] = {1, 0, 0}; s.vel[1] = {-1, 0, 0}; }
    else              { s.pos[0] = {0.5, 3.0, 3.0}; s.pos[1] = {4.5, 3.0, 3.0}; s.vel[0] = {-1, 0, 0}; s.vel[1] = {1, 0, 0}; }
    s.assign_cells();
    s.rebuild_events();
    double dmin = 1e9;
    double tcoll = -1;
    for (int k = 1; k <= 200; ++k) {
      double t = 0.005 * k;
      uint64_t before = s.ncoll;
      s.advance_to(t);
      if (s.ncoll > before && tcoll < 0) tcoll = t;
      Vec3 d = s.pos_at(0, t) - s.pos_at(1, t);
      s.minimg_full(d);
      dmin = std::min(dmin, std::sqrt(dot(d, d)));
    }
    Vec3 d = s.pos_at(0, 1.0) - s.pos_at(1, 1.0);
    s.minimg_full(d);
    const char* nm = variant == 0 ? "head-on" : "head-on across periodic face";
    char buf[128];
    std::snprintf(buf, sizeof buf, "%s: exactly one collision", nm);
    check(s.ncoll == 1, buf, (double)s.ncoll);
    std::snprintf(buf, sizeof buf, "%s: velocities swapped (|v0+1|+|v1-1|)", nm);
    double vs = variant == 0 ? std::fabs(s.vel[0].x + 1) + std::fabs(s.vel[1].x - 1) : std::fabs(s.vel[0].x - 1) + std::fabs(s.vel[1].x + 1);
    check(vs < 1e-12, buf, vs);
    std::snprintf(buf, sizeof buf, "%s: separation back to 2 at t=1", nm);
    check(std::fabs(std::sqrt(dot(d, d)) - 2.0) < 1e-12, buf, std::sqrt(dot(d, d)));
    std::snprintf(buf, sizeof buf, "%s: contact distance error at collision", nm);
    check(s.max_contact_err < 1e-12, buf, s.max_contact_err);
    std::snprintf(buf, sizeof buf, "%s: min sampled distance >= 1 (1 - dmin)", nm);
    check(dmin >= 1.0 - 1e-12, buf, 1.0 - dmin);
  }
  // 4) random two-body collisions: energy, momentum, contact distance, separating afterwards
  {
    std::mt19937_64 rng(7);
    std::normal_distribution<double> g(0, 1);
    std::uniform_real_distribution<double> u(0, 1);
    double worst_contact = 0, worst_E = 0, worst_P = 0, worst_sep = 0;
    int ncollided = 0;
    for (int trial = 0; trial < 200; ++trial) {
      Sim s;
      s.setup_box(2, 8.0);
      // relative geometry: j at origin-ish, i approaching with impact parameter b<1
      Vec3 n{g(rng), g(rng), g(rng)};
      n = n * (1.0 / std::sqrt(dot(n, n)));
      Vec3 perp{g(rng), g(rng), g(rng)};
      perp -= n * dot(perp, n);
      perp = perp * (1.0 / std::sqrt(dot(perp, perp)));
      double b = 0.95 * u(rng);
      double vmag = 0.5 + 2 * u(rng);
      Vec3 c{4 + 3 * (u(rng) - 0.5), 4 + 3 * (u(rng) - 0.5), 4 + 3 * (u(rng) - 0.5)};
      s.pos[1] = c;
      s.pos[0] = c + n * 2.0 + perp * b;
      Vec3 vcm{g(rng) * 0.3, g(rng) * 0.3, g(rng) * 0.3};
      s.vel[0] = vcm - n * vmag;
      s.vel[1] = vcm + n * (vmag * u(rng));
      s.assign_cells();
      s.rebuild_events();
      double E0 = s.kinetic();
      Vec3 P0 = s.momentum();
      s.advance_to(3.0 / vmag);
      if (s.ncoll == 1) {
        ++ncollided;
        worst_contact = std::max(worst_contact, s.max_contact_err);
        worst_E = std::max(worst_E, std::fabs(s.kinetic() - E0) / E0);
        Vec3 dP = s.momentum() - P0;
        worst_P = std::max(worst_P, std::sqrt(dot(dP, dP)));
        Vec3 d = s.pos_at(0, s.tnow) - s.pos_at(1, s.tnow);
        s.minimg_full(d);
        worst_sep = std::max(worst_sep, 1.0 - std::sqrt(dot(d, d)));
      }
    }
    check(ncollided >= 150, "random 2-body: number of trials that collided (>=150/200)", ncollided);
    check(worst_contact < 1e-10, "random 2-body: max contact distance error", worst_contact);
    check(worst_E < 1e-12, "random 2-body: max relative KE change", worst_E);
    check(worst_P < 1e-12, "random 2-body: max |dP|", worst_P);
    check(worst_sep < 1e-12, "random 2-body: max overlap after collision (1 - r)", worst_sep);
  }
  // 5) many-body: N=125 at eta=0.35 for 1 tau, brute-force overlap check every 0.05 tau
  {
    Sim s;
    double L = std::cbrt(125 * PI / (6 * 0.35));
    s.setup_box(125, L);
    s.init_lattice(3);
    double E0 = s.kinetic();
    double worst = 0;
    for (int k = 1; k <= 20; ++k) {
      s.advance_to(0.05 * k);
      worst = std::max(worst, 1.0 - std::sqrt(s.min_dist2_bruteforce()));
    }
    check(worst <= 1e-12, "N=125 EDMD 1 tau: max brute-force overlap (1 - dmin)", worst);
    check(std::fabs(s.kinetic() - E0) / E0 < 1e-12, "N=125 EDMD 1 tau: relative KE change", std::fabs(s.kinetic() - E0) / E0);
    Vec3 P = s.momentum();
    check(std::sqrt(dot(P, P)) < 1e-10, "N=125 EDMD 1 tau: |total momentum|", std::sqrt(dot(P, P)));
    check(s.ncoll > 125 * 10 * 0.5 / 2, "N=125 EDMD 1 tau: collisions happened (>~300)", (double)s.ncoll);
  }
  // 6) bound +/- pair bouncing under Coulomb attraction (collision Verlet): energy error must be
  //    second order with the physical-velocity rule and much smaller than the naive rule
  {
    auto run_pair = [&](bool phys, double dt) {
      Sim s;
      s.setup_box(2, 6.0);
      s.phys_collision = phys;
      s.rng.seed(1);
      s.pos[0] = {2.0, 3.0, 3.0}; s.pos[1] = {3.7, 3.0, 3.0};
      s.vel[0] = {0.3, 0.2, 0.0}; s.vel[1] = {-0.3, -0.2, 0.0};
      s.charge[0] = +1; s.charge[1] = -1; s.ions = {0, 1};
      s.assign_cells();
      s.rebuild_events();
      Coulomb cb;
      cb.setup(Coulomb::EWALD, 6.0, 5.0);
      std::vector<Vec3> r, F;
      std::vector<int> q;
      s.ion_charges(q);
      s.ion_positions(s.tnow, r);
      double PE = cb.compute(r, q, F);
      double E0 = s.kinetic() + PE, emax = 0;
      long nsteps = (long)std::llround(20.0 / dt);
      for (long k = 1; k <= nsteps; ++k) {
        double h = k * dt - s.tnow;
        s.kick_start(F, h);
        s.advance_to(k * dt);
        s.ion_positions(s.tnow, r);
        PE = cb.compute(r, q, F);
        s.kick_end(F, h);
        emax = std::max(emax, std::fabs(s.kinetic() + PE - E0));
      }
      return std::make_pair(emax / std::fabs(E0), (double)s.ncoll);
    };
    auto p1 = run_pair(true, 0.005), p2 = run_pair(true, 0.0025), n1 = run_pair(false, 0.005);
    std::printf("  bound pair 20 tau: collisions %.0f; max|dE|/|E0|: phys dt=0.005 %.3e, phys dt=0.0025 %.3e, naive dt=0.005 %.3e\n",
                p1.second, p1.first, p2.first, n1.first);
    check(p1.second >= 10, "bound pair: collisions occurred (>=10)", p1.second);
    check(p1.first < 3e-3, "bound pair: phys rule max rel. energy error (dt=0.005, |E0|~3, a_rel=10)", p1.first);
    check(p2.first < 0.5 * p1.first, "bound pair: phys rule error shrinks >=2x when dt halves", p2.first / p1.first);
    check(n1.first > 3 * p1.first, "bound pair: naive rule error is >3x the phys rule error", n1.first / p1.first);
  }
  std::printf("selftest: %d failure(s)\n", fails);
  return fails ? 1 : 0;
}

// ---------------------------------------------------------------------------
// Production driver
// ---------------------------------------------------------------------------
struct Diag {
  std::vector<double> t, KE, PE, E, T;
};

int run(const Options& o) {
  const double t_start = wall_now();
  const double L = std::cbrt(o.N * PI / (6 * o.eta));
  const int nion = o.nplus + o.nminus;
  if (nion > o.N) { std::fprintf(stderr, "error: nplus+nminus > N\n"); return 2; }
  const bool use_cv = (o.integrator == "cv");
  if (nion > 0 && use_cv && o.cv_q2 > 0.5 * L) {
    std::fprintf(stderr, "error: --cv-q2 %g exceeds L/2 = %g (minimum image)\n", o.cv_q2, 0.5 * L);
    return 2;
  }
  Sim s;
  s.setup_box(o.N, L);
  s.phys_collision = (o.integrator == "kdk-phys");
  s.init_lattice(o.seed);
  std::fprintf(stderr, "hsdmd %s: N=%d eta=%.4f L=%.6f ncell=%d ions=%d (+%d/-%d) coulomb=%s integrator=%s seed=%llu\n",
               HSDMD_VERSION, o.N, o.eta, L, s.ncell, nion, o.nplus, o.nminus, o.coulomb.c_str(),
               o.integrator.c_str(), (unsigned long long)o.seed);

  FILE* fxyz = nullptr;
  FILE* fdiag = nullptr;
  if (!o.xyz.empty()) { fxyz = std::fopen(o.xyz.c_str(), "w"); if (!fxyz) { std::perror(o.xyz.c_str()); return 2; } }
  if (!o.diag.empty()) { fdiag = std::fopen(o.diag.c_str(), "w"); if (!fdiag) { std::perror(o.diag.c_str()); return 2; } }
  if (fdiag) std::fprintf(fdiag, "time,KE,PE,Etot,T_inst,n_collisions,min_dist,n_pairs_pm_1p2\n");

  double next_progress = 1.0;
  auto progress = [&](const char* stage, double PE) {
    if (o.quiet) return;
    if (s.tnow + 1e-9 >= next_progress) {
      std::fprintf(stderr, "  [%s] t=%7.3f coll=%llu cross=%llu heap=%zu KE=%.4f PE=%.4f wall=%.1fs\n", stage, s.tnow,
                   (unsigned long long)s.ncoll, (unsigned long long)s.ncross, s.heap.size(), s.kinetic(), PE,
                   wall_now() - t_start);
      next_progress = std::floor(s.tnow + 1e-9) + 1.0;
    }
  };

  // ---------------- stage A: neutral equilibration (thermostat every 0.1 tau)
  {
    double tend = s.tnow + o.t_equil_neutral;
    // floor (not round): llround overshoots tend for e.g. 0.75/0.1 and the trailing advance_to(tend)
    // would then run the clock backwards; the remainder (< one block) is covered by advance_to(tend)
    int nblocks = (int)std::floor(o.t_equil_neutral / o.thermostat_interval + 1e-9);
    double t0 = s.tnow;
    for (int b = 1; b <= nblocks; ++b) {
      s.advance_to(std::min(t0 + b * o.thermostat_interval, tend));
      s.sync_all(s.tnow);
      s.rescale_velocities();
      s.rebuild_events();
      progress("equil-neutral", 0.0);
    }
    s.advance_to(tend);
  }
  const double wall_A = wall_now() - t_start;
  const uint64_t coll_A = s.ncoll;

  // ---------------- stage B: charges + charged equilibration
  Coulomb cb;
  std::vector<Vec3> rion, Fion;
  std::vector<int> qion;
  double PE = 0;
  double dt_eff = o.dt;
  int nsub = 1;
  if (nion > 0) {
    s.assign_charges(o.nplus, o.nminus);
    cb.setup(o.coulomb == "ewald" ? Coulomb::EWALD : Coulomb::DSF, L, o.bjerrum);
    s.ion_charges(qion);
    nsub = std::max(1, (int)std::llround(o.dt_frame / o.dt));
    dt_eff = o.dt_frame / nsub;
    std::fprintf(stderr, "  charged: dt=%.6g (dt_frame/dt=%d), alpha=%.4f rc=%.4f kvecs=%zu\n", dt_eff, nsub,
                 cb.alpha, cb.rc, cb.kvecs.size());
    if (use_cv) {
      s.cv_begin(o.bjerrum, o.cv_q1, o.cv_q2);
      std::fprintf(stderr, "  collision Verlet: q1=%g q2=%g, %zu ions accelerated at start, vmin=%g\n", o.cv_q1, o.cv_q2,
                   s.Alist.size(), s.cv_vmin);
    }
  }
  // one step of length h ending exactly at t_target; Fion holds the total Coulomb forces at the current
  // (synchronized) positions on entry and on exit, PE the full Coulomb energy at the exit positions.
  //  cv:  half kick with F_2 = F - F_1 -> exact H_1 flow (collisions + Verlet sub-steps with F_1) -> half kick
  //  kdk: half kick with F -> exact hard-sphere flow -> half kick (HLL eq. 9, "naive splitting")
  auto verlet_step_to = [&](double t_target) {
    const double h = t_target - s.tnow;
    if (use_cv) {
      s.cv_kick_start(Fion, h);
      s.advance_to(t_target);
      s.cv_finish_step();
      s.ion_positions(s.tnow, rion);
      PE = cb.compute(rion, qion, Fion);
      s.cv_kick_end(Fion, h);
    } else {
      s.kick_start(Fion, h);
      s.advance_to(t_target);
      s.ion_positions(s.tnow, rion);
      PE = cb.compute(rion, qion, Fion);
      s.kick_end(Fion, h);
    }
  };
  if (nion > 0) {
    s.sync_all(s.tnow);
    s.ion_positions(s.tnow, rion);
    PE = cb.compute(rion, qion, Fion);
    int steps_per_block = std::max(1, (int)std::llround(o.thermostat_interval / dt_eff));
    long nsteps = (long)std::llround(o.t_equil_charged / dt_eff);
    const double tB0 = s.tnow;
    for (long k = 1; k <= nsteps; ++k) {
      verlet_step_to(tB0 + k * dt_eff);
      if (k % steps_per_block == 0) {
        s.sync_all(s.tnow);
        s.rescale_velocities();
        s.rebuild_events();
      }
      progress("equil-charged", PE);
    }
  }
  const double wall_B = wall_now() - t_start - wall_A;

  // ---------------- stage C: production (NVE)
  const double t_prod0 = s.tnow;
  const uint64_t coll_prod0 = s.ncoll;
  s.virial = 0;
  const int nframes = (int)std::floor(o.t_run / o.dt_frame + 1e-9) + 1;
  // output order: neutrals, then + ions, then - ions (frame-invariant ids 1..N). With this order
  // OVITO's first-appearance numeric type ids coincide with FORMAT.md (HS=1, POS=2, NEG=3).
  std::vector<int> order;
  order.reserve(o.N);
  for (int i = 0; i < o.N; ++i) if (s.charge[i] == 0) order.push_back(i);
  for (int i = 0; i < o.N; ++i) if (s.charge[i] > 0) order.push_back(i);
  for (int i = 0; i < o.N; ++i) if (s.charge[i] < 0) order.push_back(i);
  Diag dg;
  double min_dist_prod = std::numeric_limits<double>::infinity();
  std::vector<char> buf;
  buf.reserve((size_t)o.N * 80 + 256);

  auto write_frame = [&](int k) {
    const double t = s.tnow;
    const double tframe = k * o.dt_frame;
    double KE = s.kinetic();
    double PEf = 0;
    if (nion > 0) PEf = PE;  // positions synced at the step end
    double E = KE + PEf;
    double T = 2 * KE / (3.0 * (o.N - 1));
    double md = s.min_dist_celllist(t);
    min_dist_prod = std::min(min_dist_prod, md);
    int npm = nion > 0 ? s.count_pm_pairs(t, 1.2 * SIGMA) : 0;
    dg.t.push_back(tframe); dg.KE.push_back(KE); dg.PE.push_back(PEf); dg.E.push_back(E); dg.T.push_back(T);
    if (fdiag)
      std::fprintf(fdiag, "%.6f,%.12g,%.12g,%.12g,%.10g,%llu,%.12f,%d\n", tframe, KE, PEf, E, T,
                   (unsigned long long)(s.ncoll - coll_prod0), md, npm);
    if (fxyz) {
      buf.clear();
      char line[160];
      int n = std::snprintf(line, sizeof line, "%d\n", o.N);
      buf.insert(buf.end(), line, line + n);
      n = std::snprintf(line, sizeof line,
                        "Lattice=\"%.10f 0 0 0 %.10f 0 0 0 %.10f\" Properties=species:S:1:pos:R:3:id:I:1:type:I:1:charge:R:1 "
                        "Time=%.6f Frame=%d pbc=\"T T T\"\n",
                        L, L, L, tframe, k);
      buf.insert(buf.end(), line, line + n);
      for (int oi = 0; oi < o.N; ++oi) {
        const int i = order[oi];
        Vec3 p = s.pos_at(i, t);
        s.wrap(p);
        if (p.x >= L) p.x = 0;
        if (p.y >= L) p.y = 0;
        if (p.z >= L) p.z = 0;
        int q = s.charge[i];
        const char* sp = q == 0 ? "HS" : (q > 0 ? "POS" : "NEG");
        int ty = q == 0 ? 1 : (q > 0 ? 2 : 3);
        n = std::snprintf(line, sizeof line, "%s %.10f %.10f %.10f %d %d %d\n", sp, p.x, p.y, p.z, oi + 1, ty, q);
        buf.insert(buf.end(), line, line + n);
      }
      std::fwrite(buf.data(), 1, buf.size(), fxyz);
    }
  };

  if (nion > 0) {
    // F is consistent with the current (synced) positions from the end of stage B
    s.sync_all(s.tnow);
    s.ion_positions(s.tnow, rion);
    PE = cb.compute(rion, qion, Fion);
    write_frame(0);
    for (int k = 1; k < nframes; ++k) {
      for (int sub = 0; sub < nsub; ++sub)  // integer step grid: no accumulation of roundoff
        verlet_step_to(t_prod0 + ((double)(k - 1) * nsub + sub + 1) * dt_eff);
      write_frame(k);
      progress("production", PE);
    }
  } else {
    write_frame(0);
    for (int k = 1; k < nframes; ++k) {
      s.advance_to(t_prod0 + k * o.dt_frame);
      write_frame(k);
      progress("production", 0.0);
    }
  }
  const double t_prod = s.tnow - t_prod0;
  const double wall_C = wall_now() - t_start - wall_A - wall_B;
  const double wall_total = wall_now() - t_start;
  if (fxyz) std::fclose(fxyz);
  if (fdiag) std::fclose(fdiag);

  // ---------------- summary
  double Tmean = 0;
  for (double T : dg.T) Tmean += T;
  Tmean /= (double)dg.T.size();
  const uint64_t ncoll_prod = s.ncoll - coll_prod0;
  const bool have_prod = t_prod > 0;  // t_run < dt_frame gives a single frame: rates/Z undefined
  const double Z = have_prod ? 1.0 + s.virial / (3.0 * o.N * Tmean * t_prod) : 0.0;
  const double Zcs = carnahan_starling_Z(o.eta);
  const double rate = have_prod ? 2.0 * (double)ncoll_prod / ((double)o.N * t_prod) : 0.0;
  const double E0 = dg.E.front(), Eend = dg.E.back();
  double Emax_dev = 0, Emean = 0;
  for (double E : dg.E) { Emax_dev = std::max(Emax_dev, std::fabs(E - E0)); Emean += E; }
  Emean /= (double)dg.E.size();
  double Erms = 0;
  for (double E : dg.E) Erms += (E - Emean) * (E - Emean);
  Erms = std::sqrt(Erms / (double)dg.E.size());
  // linear fit slope of E(t)
  double st = 0, sE = 0, stt = 0, stE = 0;
  for (size_t k = 0; k < dg.t.size(); ++k) { st += dg.t[k]; sE += dg.E[k]; stt += dg.t[k] * dg.t[k]; stE += dg.t[k] * dg.E[k]; }
  double nn = (double)dg.t.size();
  double slope = (nn * stE - st * sE) / std::max(1e-300, nn * stt - st * st);
  Vec3 P = s.momentum();
  double Pabs = std::sqrt(dot(P, P));
  double KEmean = 0;
  for (double k : dg.KE) KEmean += k;
  KEmean /= (double)dg.KE.size();
  double PEmean = 0;
  for (double p : dg.PE) PEmean += p;
  PEmean /= (double)dg.PE.size();

  std::fprintf(stderr,
               "done: coll(prod)=%llu rate=%.3f/particle/tau (Enskog %.3f) Z=%.4f (CS %.4f, %.2f%%) "
               "E0=%.6f Eend=%.6f drift_rel=%.3e max_rel=%.3e |P|=%.2e min_dist=%.12f contact_err=%.2e wall=%.2fs\n",
               (unsigned long long)ncoll_prod, rate, enskog_rate(o.eta), Z, Zcs, 100 * (Z / Zcs - 1), E0, Eend,
               (Eend - E0) / std::fabs(E0), Emax_dev / std::fabs(E0), Pabs, min_dist_prod, s.max_contact_err, wall_total);

  if (!have_prod)
    std::fprintf(stderr, "warning: t_run (%g) < dt_frame (%g): single frame, Z/collision rate undefined (written as null)\n",
                 o.t_run, o.dt_frame);

  if (!o.summary.empty()) {
    FILE* f = std::fopen(o.summary.c_str(), "w");
    if (!f) { std::perror(o.summary.c_str()); return 2; }
    // JSON has no nan/inf: non-finite values are written as null (printf would emit "-nan")
    char jbuf[64];
    auto jnum = [&](double x, const char* fmt) -> const char* {
      if (!std::isfinite(x)) return "null";
      std::snprintf(jbuf, sizeof jbuf, fmt, x);
      return jbuf;
    };
    std::fprintf(f, "{\n");
    std::fprintf(f, "  \"name\": \"%s\",\n", o.name.c_str());
    std::fprintf(f, "  \"N\": %d,\n  \"eta\": %.10g,\n  \"L\": %.12g,\n  \"kT\": 1.0,\n", o.N, o.eta, L);
    std::fprintf(f, "  \"nplus\": %d,\n  \"nminus\": %d,\n  \"bjerrum\": %.10g,\n", o.nplus, o.nminus, o.bjerrum);
    std::fprintf(f, "  \"coulomb\": \"%s\",\n", nion > 0 ? o.coulomb.c_str() : "none");
    {
      char integ[256];
      if (nion == 0) std::snprintf(integ, sizeof integ, "EDMD");
      else if (use_cv) std::snprintf(integ, sizeof integ, "collision Verlet (HLL 2000), q1=%g, q2=%g", o.cv_q1, o.cv_q2);
      else if (o.integrator == "kdk-phys") std::snprintf(integ, sizeof integ, "EDMD + kick-drift-kick (physical-velocity collision rule)");
      else std::snprintf(integ, sizeof integ, "EDMD + kick-drift-kick (naive splitting, HLL eq. 9)");
      std::fprintf(f, "  \"integrator\": \"%s\",\n", integ);
      std::fprintf(f, "  \"integrator_id\": \"%s\",\n", nion > 0 ? o.integrator.c_str() : "edmd");
      std::fprintf(f, "  \"cv_q1\": %.10g,\n  \"cv_q2\": %.10g,\n  \"cv_vmin\": %.6g,\n", o.cv_q1, o.cv_q2, s.cv_vmin);
      std::fprintf(f, "  \"n_verlet_substeps\": %llu,\n  \"n_quartic_solves\": %llu,\n  \"n_cv_vmin_bias\": %llu,\n  \"n_cv_grazing\": %llu,\n",
                   (unsigned long long)s.n_substeps, (unsigned long long)s.n_quartic, (unsigned long long)s.n_cv_vmin,
                   (unsigned long long)s.n_cv_grazing);
    }
    std::fprintf(f, "  \"collision_rule\": \"%s\",\n  \"n_phys_separating\": %llu,\n  \"n_drift_fix\": %llu,\n",
                 nion == 0 ? "exact" : (use_cv ? "cv" : (o.integrator == "kdk-phys" ? "phys" : "naive")),
                 (unsigned long long)s.n_phys_separating, (unsigned long long)s.n_drift_fix);
    std::fprintf(f, "  \"dt\": %.12g,\n  \"dt_requested\": %.12g,\n  \"steps_per_frame\": %d,\n", nion > 0 ? dt_eff : 0.0, o.dt, nion > 0 ? nsub : 0);
    std::fprintf(f, "  \"dt_frame\": %.12g,\n  \"t_equil_neutral\": %.10g,\n  \"t_equil_charged\": %.10g,\n  \"t_run\": %.10g,\n",
                 o.dt_frame, o.t_equil_neutral, nion > 0 ? o.t_equil_charged : 0.0, o.t_run);
    std::fprintf(f, "  \"n_frames\": %d,\n  \"seed\": %llu,\n", nframes, (unsigned long long)o.seed);
    std::fprintf(f, "  \"ncell\": %d,\n  \"n_collisions\": %llu,\n  \"n_collisions_total\": %llu,\n  \"n_collisions_equil_neutral\": %llu,\n",
                 s.ncell, (unsigned long long)ncoll_prod, (unsigned long long)s.ncoll, (unsigned long long)coll_A);
    std::fprintf(f, "  \"n_cell_crossings_total\": %llu,\n  \"n_events_processed\": %llu,\n  \"n_stale_events\": %llu,\n  \"n_grazing_skipped\": %llu,\n",
                 (unsigned long long)s.ncross, (unsigned long long)s.nevents, (unsigned long long)s.nstale, (unsigned long long)s.nskipped_grazing);
    std::fprintf(f, "  \"collision_rate_per_particle_per_tau\": %s,\n", have_prod ? jnum(rate, "%.6g") : "null");
    std::fprintf(f, "  \"enskog_rate_per_particle_per_tau\": %s,\n", jnum(enskog_rate(o.eta), "%.6g"));
    std::fprintf(f, "  \"Z_virial\": %s,\n", have_prod ? jnum(Z, "%.8g") : "null");
    std::fprintf(f, "  \"Z_CS\": %s,\n", jnum(Zcs, "%.8g"));
    std::fprintf(f, "  \"Z_rel_dev\": %s,\n", have_prod ? jnum(Z / Zcs - 1, "%.6g") : "null");
    std::fprintf(f, "  \"T_mean\": %s,\n", jnum(Tmean, "%.10g"));
    std::fprintf(f, "  \"KE_mean\": %s,\n", jnum(KEmean, "%.10g"));
    std::fprintf(f, "  \"PE_mean\": %s,\n", jnum(PEmean, "%.10g"));
    std::fprintf(f, "  \"E0\": %s,\n", jnum(E0, "%.12g"));
    std::fprintf(f, "  \"E_end\": %s,\n", jnum(Eend, "%.12g"));
    std::fprintf(f, "  \"energy_drift_rel\": %s,\n", jnum((Eend - E0) / std::fabs(E0), "%.6g"));
    std::fprintf(f, "  \"energy_max_dev_rel\": %s,\n", jnum(Emax_dev / std::fabs(E0), "%.6g"));
    std::fprintf(f, "  \"energy_rms_rel\": %s,\n", jnum(Erms / std::fabs(E0), "%.6g"));
    std::fprintf(f, "  \"energy_slope_per_tau_rel\": %s,\n", jnum(slope / std::fabs(E0), "%.6g"));
    std::fprintf(f, "  \"momentum_abs_final\": %s,\n", jnum(Pabs, "%.6g"));
    std::fprintf(f, "  \"min_dist_production\": %s,\n", jnum(min_dist_prod, "%.15g"));
    std::fprintf(f, "  \"max_contact_err\": %s,\n", jnum(s.max_contact_err, "%.6g"));
    std::fprintf(f, "  \"wall_seconds\": %.3f,\n  \"wall_seconds_equil_neutral\": %.3f,\n  \"wall_seconds_equil_charged\": %.3f,\n  \"wall_seconds_production\": %.3f,\n",
                 wall_total, wall_A, wall_B, wall_C);
    if (nion > 0)
      std::fprintf(f, "  \"ewald_alpha\": %.10g,\n  \"coulomb_rc\": %.10g,\n  \"ewald_kmax\": %d,\n  \"n_kvectors_halfspace\": %zu,\n",
                   cb.alpha, cb.rc, cb.mode == Coulomb::EWALD ? cb.kmax : 0, cb.kvecs.size());
    std::fprintf(f, "  \"hsdmd_version\": \"%s\"\n", HSDMD_VERSION);
    std::fprintf(f, "}\n");
    std::fclose(f);
  }
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  Options o = parse(argc, argv);
  if (o.selftest) return selftest();
  return run(o);
}
