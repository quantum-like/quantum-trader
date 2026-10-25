#!/usr/bin/env python
"""Dependency-free fallback renderer (numpy + Pillow only) with the SAME CLI as render_ovito.py.

Used automatically by the run scripts when `import ovito` fails (e.g. a Mac without a working
Vulkan/Qt offscreen stack). Produces the same movie: perspective camera orbiting the periodic box,
opaque CPK spheres (gray neutrals, red +, blue -), soft key+fill Blinn-Phong lighting with a
neighbour-count ambient-occlusion cue, analytic edge antialiasing, faint depth-tested box edges,
label/time/legend overlay, H.264 mp4 via ffmpeg.

    python render/render_numpy.py --traj traj/hs_neutral.npz --name hs_neutral --jobs 4

Per-sphere bounding-box rasterisation: for every sphere (sorted back to front) the pixel rays of its
screen bounding box are intersected analytically (ray-sphere), depth tested against a z-buffer and
shaded; the silhouette gets a sub-pixel coverage value from the ray-centre distance so edges are
antialiased.  --style xray composites the 85 % transparent neutrals over the opaque ions.
"""
import argparse
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (BG, CELL_COLOR, COLORS, RADIUS, XRAY_TRANSPARENCY, Camera, build_parser,  # noqa: E402
                    check_tools, default_label, finish_movie, frame_indices, n_output_frames,
                    output_pairs, render_frames, resolve_paths)
from trajio import load_traj  # noqa: E402


def _unit(v):
    v = np.asarray(v, np.float32)
    return v / np.linalg.norm(v)


class SphereRasterizer:
    """Camera-space convention: x right, y up, z = forward (depth)."""

    def __init__(self, cam, bg, style, quality):
        self.cam = cam
        self.W, self.H = cam.width, cam.height
        th = math.tan(cam.fov / 2.0)
        tw = th * cam.aspect
        self.th, self.tw = th, tw
        xs = (((np.arange(self.W) + 0.5) / self.W) * 2.0 - 1.0) * tw
        ys = (1.0 - ((np.arange(self.H) + 0.5) / self.H) * 2.0) * th
        X, Y = np.meshgrid(xs.astype(np.float32), ys.astype(np.float32))
        D = np.stack([X, Y, np.ones_like(X)], -1)
        D /= np.linalg.norm(D, axis=-1, keepdims=True)
        self.dirs = np.ascontiguousarray(D, dtype=np.float32)          # (H,W,3)
        self.footprint = 2.0 * th / self.H                               # world size of 1 px at depth 1
        self.bg = np.array(BG[bg], np.float32)
        self.line_color = np.array(CELL_COLOR[bg], np.float32)
        self.style = style
        self.quality = quality
        # lights (camera space, pointing from the surface towards the light)
        self.key_dir = _unit([-0.42, 0.55, -0.72])
        self.fill_dir = _unit([0.62, -0.28, -0.55])
        self.key_int, self.fill_int = 0.72, 0.22
        self.ambient0 = 0.45
        self.spec_int, self.shininess = 0.30, 40.0
        self.ao_strength = 0.55 if quality == "final" else 0.0
        self.line_alpha = 1.0

    # ---------------------------------------------------------------- helpers
    def ao_factor(self, pos, L):
        """Cheap ambient-occlusion cue: neighbours within 1.5 sigma (minimum image) darken the ambient term."""
        if self.ao_strength <= 0.0:
            return np.ones(len(pos), np.float32)
        p = pos.astype(np.float32)
        cnt = np.zeros(len(p), np.int32)
        chunk = 256
        for i0 in range(0, len(p), chunk):
            d = p[i0:i0 + chunk, None, :] - p[None, :, :]
            d -= L * np.rint(d / L)
            r2 = (d * d).sum(-1)
            cnt[i0:i0 + chunk] = (r2 < 2.25).sum(1) - 1
        return (1.0 - self.ao_strength * np.minimum(cnt, 12) / 12.0).astype(np.float32)

    def shade(self, n, d, base, ao):
        """n: (h,w,3) unit normals, d: (h,w,3) ray dirs, base: (3,), ao: scalar -> (h,w,3) colour."""
        ndk = np.clip(n @ self.key_dir, 0.0, None)
        ndf = np.clip(n @ self.fill_dir, 0.0, None)
        half = self.key_dir - d                                   # L + V with V = -d
        half /= np.linalg.norm(half, axis=-1, keepdims=True)
        spec = np.clip((n * half).sum(-1), 0.0, None) ** self.shininess * self.spec_int
        diffuse = self.ambient0 * ao + self.key_int * ndk + self.fill_int * ndf
        col = base[None, None, :] * diffuse[..., None] + spec[..., None]
        return col

    # ---------------------------------------------------------------- frame
    def render(self, pos_world, types, pose, L):
        W, H, R = self.W, self.H, RADIUS
        th, tw = self.th, self.tw
        img = np.empty((H, W, 3), np.float32)
        img[:] = self.bg
        zbuf = np.full((H, W), np.inf, np.float32)
        M = np.stack([pose["right"], pose["up"], pose["dir"]]).astype(np.float32)   # rows
        C = ((pos_world.astype(np.float32) - pose["pos"].astype(np.float32)) @ M.T)  # (N,3) camera space
        ao = self.ao_factor(pos_world, L)
        base = np.array([COLORS[int(t)] for t in (1, 2, 3)], np.float32)
        alpha_type = np.ones(4, np.float32)
        if self.style == "xray":
            alpha_type[1] = 1.0 - XRAY_TRANSPARENCY
        order = np.argsort(-C[:, 2])                      # back to front
        dirs = self.dirs
        R2 = R * R
        for i in order:
            c = C[i]
            cz = float(c[2])
            if cz <= R + 1e-3:
                continue
            px = (float(c[0]) / (cz * tw) + 1.0) * 0.5 * W
            py = (1.0 - float(c[1]) / (cz * th)) * 0.5 * H
            rp = R * (0.5 * H) / (th * (cz - R)) * 1.12 + 2.5
            x0, x1 = max(0, int(px - rp)), min(W, int(px + rp) + 2)
            y0, y1 = max(0, int(py - rp)), min(H, int(py + rp) + 2)
            if x0 >= x1 or y0 >= y1:
                continue
            d = dirs[y0:y1, x0:x1]
            cd = d @ c
            b2 = float(c @ c) - cd * cd
            fp = self.footprint * cz
            b = np.sqrt(np.maximum(b2, 0.0))
            cov = np.clip((R - b) / fp + 0.5, 0.0, 1.0)
            m = cov > 0.0
            if not m.any():
                continue
            t = cd - np.sqrt(np.clip(R2 - b2, 0.0, None))
            zb = zbuf[y0:y1, x0:x1]
            m &= t < zb
            if not m.any():
                continue
            n = t[..., None] * d - c
            n /= np.linalg.norm(n, axis=-1, keepdims=True)
            tp = int(types[i])
            col = self.shade(n, d, base[tp - 1], float(ao[i]))
            a = (cov * alpha_type[tp])[..., None]
            sub = img[y0:y1, x0:x1]
            blended = sub * (1.0 - a) + col * a
            sub[m] = blended[m]
            if alpha_type[tp] >= 1.0:
                mz = m & (cov >= 0.5)
                zb[mz] = t[mz]
        self.draw_box_edges(img, zbuf, pose, L)
        return (np.clip(img, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)

    def draw_box_edges(self, img, zbuf, pose, L):
        W, H = self.W, self.H
        corners = np.array([[x, y, z] for x in (0.0, L) for y in (0.0, L) for z in (0.0, L)])
        edges = [(a, b) for a in range(8) for b in range(a + 1, 8)
                 if np.count_nonzero(corners[a] != corners[b]) == 1]
        pts = []
        for a, b in edges:
            pa, pb = corners[a], corners[b]
            pix, _ = self.cam.project(pose, np.stack([pa, pb]))
            length = np.linalg.norm(pix[1] - pix[0])
            ns = max(2, int(3.0 * length))
            s = np.linspace(0.0, 1.0, ns)[:, None]
            pts.append(pa + (pb - pa) * s)
        P = np.concatenate(pts)
        pix, depth = self.cam.project(pose, P)
        # The z-buffer stores the ray parameter t = Euclidean distance from the camera (unit ray dirs),
        # NOT the view-axis depth, so the visibility test must use the same quantity: off-axis the
        # view depth is smaller than the distance by up to ~1.7 sigma and edges would show through spheres.
        dist = np.linalg.norm(P - pose["pos"], axis=1)
        x, y = pix[:, 0], pix[:, 1]
        ok = (x >= 0) & (x < W - 1) & (y >= 0) & (y < H - 1) & (depth > 0)
        x, y, dist = x[ok], y[ok], dist[ok]
        xi, yi = np.floor(x).astype(int), np.floor(y).astype(int)
        # distance test against the nearest pixel (small bias so lines on the box faces stay visible)
        vis = dist < zbuf[np.clip(np.rint(y).astype(int), 0, H - 1), np.clip(np.rint(x).astype(int), 0, W - 1)] + 0.05
        x, y, xi, yi = x[vis], y[vis], xi[vis], yi[vis]
        fx, fy = x - xi, y - yi
        cov = np.zeros((H, W), np.float32)
        for dx, dy, w in ((0, 0, (1 - fx) * (1 - fy)), (1, 0, fx * (1 - fy)),
                          (0, 1, (1 - fx) * fy), (1, 1, fx * fy)):
            np.add.at(cov, (yi + dy, xi + dx), w)
        cov = np.clip(cov / 2.5, 0.0, 1.0) * self.line_alpha
        img *= (1.0 - cov)[..., None]
        img += cov[..., None] * self.line_color


# -------------------------------------------------------------------- driver

def add_numpy_args(p):
    p.add_argument("--jobs", type=int, default=max(1, min(4, os.cpu_count() or 1)),
                   help="worker processes (frames are split round-robin)")
    return p


def _make_renderer(args, traj, n_total):
    # nout = frames of the FULL movie so that the azimuth of output frame j is the same in every chunk
    cam = Camera(traj.L, args.width, args.height, args.fov_deg, args.elev_deg, args.azim0_deg,
                 args.orbit_deg, nout=n_total)
    rast = SphereRasterizer(cam, args.bg, args.style, args.quality)

    def render_rgb(j, k):
        return rast.render(traj.pos[k], traj.types, cam.pose(j), traj.L)

    return cam, render_rgb


def _worker(argdict, pairs, total, wid):
    args = argparse.Namespace(**argdict)
    traj = load_traj(args.traj)
    name, out, frames_dir = resolve_paths(args)
    label = default_label(traj, name) if args.label is None else args.label
    _, render_rgb = _make_renderer(args, traj, n_output_frames(traj.nframes, args.stride))
    return render_frames(args, traj, pairs, render_rgb, frames_dir, label, tag=f"[numpy w{wid}] ", total=total)


def main(argv=None):
    p = add_numpy_args(build_parser("render_numpy.py", __doc__))
    args = p.parse_args(argv)
    check_tools(args)
    t_setup = time.time()
    traj = load_traj(args.traj)
    name, out, frames_dir = resolve_paths(args)
    indices = frame_indices(traj.nframes, args.start, args.end, args.stride)
    n_total = n_output_frames(traj.nframes, args.stride)
    pairs = output_pairs(indices, args.stride)
    label = default_label(traj, name) if args.label is None else args.label
    print(f"trajectory {args.traj}: {traj.describe()}")
    print(f"name={name} frames={len(indices)} (traj {indices[0]}..{indices[-1]} stride {args.stride} -> "
          f"output frames {pairs[0][0]}..{pairs[-1][0]} of {n_total}) "
          f"size={args.width}x{args.height} style={args.style} bg={args.bg} quality={args.quality} jobs={args.jobs}")
    cam, render_rgb = _make_renderer(args, traj, n_total)
    print(cam.describe())
    print(f"setup {time.time() - t_setup:.1f}s; rendering to {frames_dir}", flush=True)
    t0 = time.time()
    if args.jobs <= 1 or len(pairs) < 2:
        render_frames(args, traj, pairs, render_rgb, frames_dir, label, tag="[numpy] ", total=n_total)
    else:
        import multiprocessing as mp
        jobs = min(args.jobs, len(pairs))
        chunks = [pairs[w::jobs] for w in range(jobs)]
        ctx = mp.get_context("spawn")        # portable (macOS default); workers re-load the trajectory
        with ctx.Pool(jobs) as pool:
            results = pool.starmap(_worker, [(vars(args), ch, n_total, w) for w, ch in enumerate(chunks)])
        n = sum(r[0] for r in results)
        wall = time.time() - t0
        if n:
            print(f"[numpy] {n} frames with {jobs} workers in {wall:.1f}s -> {wall / n:.3f} s/frame wall "
                  f"(721 frames ~{721 * wall / n / 60:.1f} min)")
    finish_movie(args, frames_dir, out, pairs, n_total)


if __name__ == "__main__":
    main()
