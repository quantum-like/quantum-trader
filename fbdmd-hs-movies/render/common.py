"""Shared pieces for render_ovito.py and render_numpy.py: CLI, colours, camera/orbit maths,
Pillow overlay, resumable frame loop, ffmpeg encode + ffprobe verification.

Only numpy + Pillow are required here.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import shutil
import subprocess
import sys
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ------------------------------------------------------------------ look (FORMAT.md)
RADIUS = 0.5
TYPE_NAMES = {1: "HS", 2: "POS", 3: "NEG"}
COLORS = {1: (0.78, 0.78, 0.80), 2: (0.85, 0.15, 0.15), 3: (0.15, 0.35, 0.90)}
BG = {"light": (0.97, 0.97, 0.98), "dark": (0.07, 0.08, 0.10)}
CELL_COLOR = {"light": (0.35, 0.35, 0.40), "dark": (0.75, 0.75, 0.80)}
TEXT_COLOR = {"light": (55, 55, 65), "dark": (215, 215, 225)}
XRAY_TRANSPARENCY = 0.85          # neutral spheres in --style xray (ions stay opaque)
CELL_LINE_WIDTH_REF = 0.035       # at L = 11.4 (scaled linearly with L)
FOV_DEG = 30.0                    # vertical field of view of the perspective camera
FIT_MARGIN = 0.05                 # 5 % frame margin kept around the box for every azimuth

FRAME_PATTERN = "frame_%05d.png"


def cell_line_width(L):
    return CELL_LINE_WIDTH_REF * (L / 11.4)


# ------------------------------------------------------------------ CLI

def build_parser(prog, description):
    p = argparse.ArgumentParser(prog=prog, description=description,
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--traj", required=True, help="traj/<name>.npz (preferred) or traj/<name>.xyz")
    p.add_argument("--name", default=None, help="movie name (default: trajectory basename)")
    p.add_argument("--out", default=None, help="output mp4 (default: <root>/movies/<name>.mp4)")
    p.add_argument("--frames-dir", default=None, help="PNG directory (default: <root>/frames/<name>)")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--style", choices=["cpk", "xray"], default="cpk",
                   help="cpk: all spheres opaque; xray: neutrals 85%% transparent, ions opaque")
    p.add_argument("--bg", choices=["light", "dark"], default="light")
    p.add_argument("--quality", choices=["draft", "final"], default="final")
    p.add_argument("--orbit-deg", type=float, default=40.0, help="total camera azimuth sweep over the movie")
    p.add_argument("--elev-deg", type=float, default=22.0, help="camera elevation above the xy plane")
    p.add_argument("--azim0-deg", type=float, default=None,
                   help="start azimuth (default 45 - orbit/2, i.e. the orbit is centred on the 3/4 view)")
    p.add_argument("--fov-deg", type=float, default=FOV_DEG, help="vertical field of view")
    p.add_argument("--start", type=int, default=0, help="first trajectory frame")
    p.add_argument("--end", type=int, default=-1, help="last trajectory frame (inclusive; -1 = last)")
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--label", default=None,
                   help="top-left caption (default: built from trajectory metadata; '' disables)")
    p.add_argument("--no-legend", action="store_true", help="hide the colour legend")
    p.add_argument("--no-overlay", action="store_true", help="no text at all")
    p.add_argument("--no-encode", action="store_true", help="only write PNG frames")
    p.add_argument("--keep-frames", action="store_true",
                   help="keep PNG frames after a verified encode (default: delete them)")
    p.add_argument("--force", action="store_true", help="re-render frames whose PNG already exists")
    p.add_argument("--crf", type=int, default=18)
    p.add_argument("--preset", default="slow", help="libx264 preset")
    return p


def resolve_paths(args):
    name = args.name or os.path.splitext(os.path.basename(args.traj))[0]
    out = args.out or os.path.join(PROJECT_ROOT, "movies", f"{name}.mp4")
    frames_dir = args.frames_dir or os.path.join(PROJECT_ROOT, "frames", name)
    return name, os.path.abspath(out), os.path.abspath(frames_dir)


def frame_indices(nframes, start, end, stride):
    if end < 0:
        end = nframes + end
    end = min(end, nframes - 1)
    start = max(0, start)
    if start > end:
        raise SystemExit(f"empty frame range: start={start} end={end} (nframes={nframes})")
    return list(range(start, end + 1, max(1, stride)))


def default_label(traj, name):
    m = traj.meta or {}
    c = traj.counts()
    N = m.get("N", traj.N)
    eta = m.get("eta")
    if eta is None:
        eta = N * math.pi / 6.0 / traj.L ** 3
    parts = [f"N = {N} hard spheres", f"\u03b7 = {eta:.2f}"]
    if c[2] or c[3]:
        parts.append(f"{c[2]} + / {c[3]} \u2212 ions")
        if m.get("bjerrum") is not None:
            parts.append(f"l_B/\u03c3 = {float(m['bjerrum']):g}")
    else:
        parts.append("no charges")
    return "   ".join(parts)


# ------------------------------------------------------------------ camera / orbit

def _unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


class Camera:
    """Perspective camera orbiting the box centre at constant elevation and constant distance.

    The distance is fitted ONCE so that the box (expanded by the sphere radius) fits inside the
    frame with `margin` for every azimuth of the orbit -> no popping / re-zooming during the movie.
    Convention: z is up; azimuth is measured in the xy plane from +x; camera_dir looks at the centre.
    """

    def __init__(self, L, width, height, fov_deg=FOV_DEG, elev_deg=22.0, azim0_deg=None,
                 orbit_deg=40.0, nout=1, margin=FIT_MARGIN, radius=RADIUS):
        self.L = float(L)
        self.center = np.full(3, self.L / 2.0)
        self.width, self.height = int(width), int(height)
        self.aspect = width / float(height)
        self.fov = math.radians(fov_deg)
        self.elev = math.radians(elev_deg)
        self.orbit = math.radians(orbit_deg)
        a0 = (45.0 - orbit_deg / 2.0) if azim0_deg is None else azim0_deg
        self.azim0 = math.radians(a0)
        self.nout = max(1, int(nout))
        self.radius = radius
        self.margin = margin
        self.distance = self.fit_distance()

    # --- geometry
    def azimuth(self, j):
        if self.nout == 1:
            return self.azim0 + 0.5 * self.orbit
        return self.azim0 + self.orbit * (j / (self.nout - 1))

    def basis(self, azim):
        ce, se = math.cos(self.elev), math.sin(self.elev)
        u = np.array([ce * math.cos(azim), ce * math.sin(azim), se])  # centre -> camera
        forward = -u
        right = _unit(np.cross(forward, [0.0, 0.0, 1.0]))
        up = np.cross(right, forward)
        return u, forward, right, up

    def corners(self):
        r = self.radius
        return np.array(list(itertools.product([-r, self.L + r], repeat=3))) - self.center

    def fit_distance(self, nsamples=91):
        tanh = math.tan(self.fov / 2.0) * (1.0 - self.margin)
        tanw = tanh * self.aspect
        P = self.corners()
        d = 0.0
        for azim in np.linspace(self.azim0, self.azim0 + self.orbit, nsamples):
            _, f, r, up = self.basis(azim)
            depth0 = P @ f
            d = max(d, float(np.max(np.abs(P @ up) / tanh - depth0)),
                    float(np.max(np.abs(P @ r) / tanw - depth0)))
        return d

    def pose(self, j):
        azim = self.azimuth(j)
        u, f, r, up = self.basis(azim)
        return {"pos": self.center + self.distance * u, "dir": f, "right": r, "up": up,
                "fov": self.fov, "azim_deg": math.degrees(azim)}

    def project(self, pose, points):
        """World points (M,3) -> pixel coords (M,2) [x right, y down] and depth (M,)."""
        rel = np.asarray(points, float) - pose["pos"]
        depth = rel @ pose["dir"]
        x = rel @ pose["right"]
        y = rel @ pose["up"]
        th = math.tan(self.fov / 2.0)
        ndc_x = x / (depth * th * self.aspect)
        ndc_y = y / (depth * th)
        px = (ndc_x + 1.0) * 0.5 * self.width
        py = (1.0 - ndc_y) * 0.5 * self.height
        return np.stack([px, py], 1), depth

    def worst_corner_margin_px(self):
        """Smallest distance (px) between any projected box corner and the frame border over the orbit."""
        P = self.corners() + self.center
        worst = float("inf")
        for j in range(self.nout):
            pix, _ = self.project(self.pose(j), P)
            worst = min(worst, pix[:, 0].min(), self.width - pix[:, 0].max(),
                        pix[:, 1].min(), self.height - pix[:, 1].max())
        return worst

    def describe(self):
        return (f"camera: fov={math.degrees(self.fov):.1f} deg (vertical), elev={math.degrees(self.elev):.1f} deg, "
                f"azimuth {math.degrees(self.azim0):.1f} -> {math.degrees(self.azim0 + self.orbit):.1f} deg over "
                f"{self.nout} frames, distance={self.distance:.3f} sigma (fixed), "
                f"worst-case corner margin {self.worst_corner_margin_px():.1f} px")


# ------------------------------------------------------------------ overlay (Pillow)

_FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/TTF/DejaVuSans.ttf",
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/Library/Fonts/Arial.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "C:/Windows/Fonts/arial.ttf",
]
_font_cache = {}


def get_font(size):
    size = int(size)
    if size in _font_cache:
        return _font_cache[size]
    font = None
    for path in _FONT_CANDIDATES:
        if os.path.exists(path):
            try:
                font = ImageFont.truetype(path, size)
                break
            except Exception:
                continue
    if font is None:
        try:
            font = ImageFont.load_default(size=size)
        except TypeError:
            font = ImageFont.load_default()
    _font_cache[size] = font
    return font


def draw_overlay(img, label, time_text, bg, legend_types=(), show_legend=True):
    """Materials-Studio-ish unobtrusive caption: label top-left, time bottom-left, legend bottom-right."""
    W, H = img.size
    pad = max(8, round(H * 0.025))
    fs = max(11, round(H * 0.028))
    font = get_font(fs)
    small = get_font(max(10, round(fs * 0.9)))
    color = TEXT_COLOR[bg]
    d = ImageDraw.Draw(img)
    if label:
        d.text((pad, pad), label, font=font, fill=color)
    if time_text:
        d.text((pad, H - pad - fs), time_text, font=font, fill=color)
    if show_legend and legend_types:
        names = {1: "hard sphere", 2: "+e", 3: "\u2212e"}
        items = [(t, names[t]) for t in (1, 2, 3) if t in legend_types]
        rad = round(fs * 0.38)
        gap = round(fs * 0.5)
        x = W - pad
        y = H - pad - fs
        # right-aligned, draw from the right end
        for t, txt in reversed(items):
            tw = d.textlength(txt, font=small)
            x -= tw
            d.text((x, y + (fs - small.size) / 2), txt, font=small, fill=color)
            x -= rad * 2 + round(fs * 0.3)
            cx, cy = x + rad, y + fs / 2
            rgb = tuple(int(round(255 * c)) for c in COLORS[t])
            d.ellipse([cx - rad, cy - rad, cx + rad, cy + rad], fill=rgb,
                      outline=(90, 90, 100) if bg == "light" else (190, 190, 200))
            x -= gap
    return img


def time_label(t):
    return f"t = {t:.2f} \u03c4"


# ------------------------------------------------------------------ frame loop

def save_png_atomic(img, path):
    tmp = path + ".tmp.png"
    img.save(tmp, format="PNG", compress_level=3)
    os.replace(tmp, path)


def render_frames(args, traj, pairs, render_rgb, frames_dir, label, tag="", total=None):
    """Resumable loop over (j, k) pairs: `render_rgb(j, k)` returns an (H,W,3) uint8 array for
    output frame j (trajectory frame k). Returns (n_rendered, mean_render_s, mean_total_s)."""
    os.makedirs(frames_dir, exist_ok=True)
    legend = {int(t) for t in np.unique(traj.types)}
    total = total or len(pairs)
    n_done = 0
    t_render = 0.0
    t_total = 0.0
    t_start = time.time()
    n_todo = sum(1 for j, _ in pairs
                 if args.force or not os.path.exists(os.path.join(frames_dir, FRAME_PATTERN % j)))
    for j, k in pairs:
        path = os.path.join(frames_dir, FRAME_PATTERN % j)
        if not args.force and os.path.exists(path):
            continue
        t0 = time.time()
        rgb = render_rgb(j, k)
        t1 = time.time()
        img = Image.fromarray(np.ascontiguousarray(rgb), "RGB")
        if not args.no_overlay:
            draw_overlay(img, label, time_label(traj.times[k]), args.bg, legend, not args.no_legend)
        save_png_atomic(img, path)
        t2 = time.time()
        n_done += 1
        t_render += t1 - t0
        t_total += t2 - t0
        if n_done == 1 or n_done % 10 == 0 or n_done == n_todo:
            avg = t_total / n_done
            eta = avg * (n_todo - n_done)
            print(f"{tag}frame {j + 1}/{total} (traj {k}, t={traj.times[k]:.3f}) "
                  f"render {t1 - t0:.2f}s  avg {avg:.2f}s/frame  ETA {eta / 60:.1f} min", flush=True)
    if n_done:
        print(f"{tag}rendered {n_done} frames in {time.time() - t_start:.1f}s: "
              f"mean {t_render / n_done:.3f} s/frame render, {t_total / n_done:.3f} s/frame incl. overlay+PNG; "
              f"721 frames would take ~{721 * t_total / n_done / 60:.1f} min", flush=True)
    else:
        print(f"{tag}all {len(pairs)} frames already present in {frames_dir} (use --force to re-render)")
    return n_done, (t_render / n_done if n_done else 0.0), (t_total / n_done if n_done else 0.0)


# ------------------------------------------------------------------ ffmpeg / ffprobe

def encode_movie(frames_dir, fps, out, crf=18, preset="slow"):
    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg not found on PATH (brew install ffmpeg / apt install ffmpeg)")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps),
           "-i", os.path.join(frames_dir, FRAME_PATTERN),
           "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",   # yuv420p needs even dimensions
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", str(crf), "-preset", preset,
           "-movflags", "+faststart", out]
    print("ffmpeg:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    return out


def ffprobe(path):
    if shutil.which("ffprobe") is None:
        print("ffprobe not found; skipping verification")
        return None
    cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0",
           "-show_entries", "stream=codec_name,width,height,pix_fmt,r_frame_rate,avg_frame_rate,nb_frames",
           "-show_entries", "format=duration,size", "-of", "json", path]
    info = json.loads(subprocess.run(cmd, check=True, capture_output=True, text=True).stdout)
    st = info["streams"][0]
    fmt = info["format"]
    num, den = st["r_frame_rate"].split("/")
    fps = float(num) / float(den)
    summary = {"file": path, "codec": st["codec_name"], "pix_fmt": st["pix_fmt"],
               "width": st["width"], "height": st["height"], "fps": fps,
               "nb_frames": int(st.get("nb_frames", 0) or 0),
               "duration_s": float(fmt["duration"]), "size_MB": int(fmt["size"]) / 1e6}
    print("ffprobe:", json.dumps(summary))
    return summary


def finish_movie(args, frames_dir, out, n_frames_expected):
    """Encode (unless --no-encode), verify with ffprobe, delete frames unless --keep-frames."""
    if args.no_encode:
        print(f"frames in {frames_dir} (encoding skipped)")
        return None
    present = sum(1 for j in range(n_frames_expected)
                  if os.path.exists(os.path.join(frames_dir, FRAME_PATTERN % j)))
    if present != n_frames_expected:
        raise SystemExit(f"only {present}/{n_frames_expected} frames present in {frames_dir}; not encoding")
    encode_movie(frames_dir, args.fps, out, args.crf, args.preset)
    info = ffprobe(out)
    ok = info is not None and info["codec"] == "h264" and info["pix_fmt"] == "yuv420p" \
        and info["nb_frames"] == n_frames_expected
    if ok and not args.keep_frames:
        for j in range(n_frames_expected):
            try:
                os.remove(os.path.join(frames_dir, FRAME_PATTERN % j))
            except FileNotFoundError:
                pass
        print(f"deleted {n_frames_expected} PNG frames from {frames_dir} (use --keep-frames to keep them)")
    elif not ok:
        print("WARNING: ffprobe verification failed; frames kept", file=sys.stderr)
    print(f"movie: {out}")
    return info
