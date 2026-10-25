#!/usr/bin/env python
"""Materials-Studio-style CPK movie renderer built on the OVITO Python module (TachyonRenderer).

    python render/render_ovito.py --traj traj/hs_charges_20p20m.npz --name hs_charges_20p20m \
        --out movies/hs_charges_20p20m.mp4 --frames-dir frames/hs_charges_20p20m \
        --width 1280 --height 720 --fps 30 --style cpk --bg light --quality final \
        --orbit-deg 40 --elev-deg 22 --label "Hard spheres + 20+/20- ions"

Design
* ONE pipeline: StaticSource(frame-0 DataCollection: cell, particle types/colours/radii,
  optional Transparency) + a Python modifier that overwrites the Position array with
  traj.pos[frame] (positions only, no per-frame import).
* Camera: perspective, constant elevation, slow azimuth orbit around the box centre at a distance
  fitted once over the whole orbit (5 % margin) -> no popping / re-zoom.
* Tachyon presets  final: AO (6 samples) + shadows + 4x AA (~1.3 s / 720p frame on 4 cores)
                   draft: no AO, no shadows, 2x AA (~0.8 s at 720p, ~0.2 s at 640x360)
* Overlay (label, time, legend) is drawn with Pillow on the rendered image, then PNG, then ffmpeg.
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (BG, CELL_COLOR, COLORS, RADIUS, TYPE_NAMES, XRAY_TRANSPARENCY, Camera,  # noqa: E402
                    build_parser, cell_line_width, default_label, finish_movie, frame_indices,
                    render_frames, resolve_paths)
from trajio import load_traj  # noqa: E402

TACHYON_PRESETS = {
    # measured on a 4-core x86 box at 1280x720, N = 1000
    ("cpk", "final"): dict(ambient_occlusion=True, ambient_occlusion_samples=6, ambient_occlusion_brightness=0.8,
                           shadows=True, antialiasing=True, antialiasing_samples=4,
                           direct_light=True, direct_light_intensity=0.9),          # ~1.2 s/frame
    ("cpk", "draft"): dict(ambient_occlusion=False, shadows=False, antialiasing=True, antialiasing_samples=2,
                           direct_light=True, direct_light_intensity=0.9),          # ~0.6 s/frame (0.17 s at 640x360)
    # Transparency (xray) rules: AO multiplies the cost ~6-10x (12 s/frame with AO on), a small
    # max_ray_recursion fragments interior ions, and AO off + shadows off makes Tachyon drop the
    # transparent spheres entirely. So: AO off, shadows ON, recursion depth 50 (default).
    ("xray", "final"): dict(ambient_occlusion=False, shadows=True, antialiasing=True, antialiasing_samples=4,
                            direct_light=True, direct_light_intensity=0.9, max_ray_recursion=50),   # ~1.8 s/frame
    ("xray", "draft"): dict(ambient_occlusion=False, shadows=True, antialiasing=True, antialiasing_samples=2,
                            direct_light=True, direct_light_intensity=0.9, max_ray_recursion=50),   # ~1.3 s/frame
}


def qimage_to_rgb(img):
    from ovito.qt_compat import QtGui
    img = img.convertToFormat(QtGui.QImage.Format.Format_RGBA8888)
    w, h, bpl = img.width(), img.height(), img.bytesPerLine()
    buf = img.constBits()
    if hasattr(buf, "setsize"):            # PyQt-style voidptr
        buf.setsize(bpl * h)
    arr = np.frombuffer(buf, np.uint8, count=bpl * h).reshape(h, bpl)[:, :w * 4].reshape(h, w, 4)
    return arr[:, :, :3].copy()


def build_data_collection(traj, style, bg):
    from ovito.data import DataCollection
    L = traj.L
    data = DataCollection()
    cell = data.create_cell([[L, 0, 0, 0], [0, L, 0, 0], [0, 0, L, 0]], pbc=(True, True, True))
    cell.vis.line_width = cell_line_width(L)
    cell.vis.rendering_color = CELL_COLOR[bg]
    p = data.create_particles(count=traj.N)
    p.create_property("Position", data=traj.pos[0].astype(np.float64))
    tprop = p.create_property("Particle Type", data=traj.types.astype(np.int32))
    for tid in (1, 2, 3):
        t = tprop.add_type_id(tid, p, name=TYPE_NAMES[tid])
        t.color = COLORS[tid]
        t.radius = RADIUS
    if style == "xray":
        alpha = np.where(traj.types == 1, XRAY_TRANSPARENCY, 0.0).astype(np.float64)
        p.create_property("Transparency", data=alpha)
    return data


def main(argv=None):
    p = build_parser("render_ovito.py", __doc__)
    p.add_argument("--ao-samples", type=int, default=None, help="override preset ambient_occlusion_samples")
    p.add_argument("--aa-samples", type=int, default=None, help="override preset antialiasing_samples")
    p.add_argument("--ao-brightness", type=float, default=None, help="override preset ambient_occlusion_brightness")
    p.add_argument("--no-shadows", action="store_true")
    args = p.parse_args(argv)

    t_setup = time.time()
    traj = load_traj(args.traj)
    name, out, frames_dir = resolve_paths(args)
    indices = frame_indices(traj.nframes, args.start, args.end, args.stride)
    label = default_label(traj, name) if args.label is None else args.label
    print(f"trajectory {args.traj}: {traj.describe()}")
    print(f"name={name} frames={len(indices)} ({indices[0]}..{indices[-1]} stride {args.stride}) "
          f"size={args.width}x{args.height} style={args.style} bg={args.bg} quality={args.quality}")

    cam = Camera(traj.L, args.width, args.height, args.fov_deg, args.elev_deg, args.azim0_deg,
                 args.orbit_deg, nout=len(indices))
    print(cam.describe())

    import ovito  # noqa: F401  (import after argparse so --help works without the Qt stack)
    from ovito.pipeline import Pipeline, StaticSource
    from ovito.vis import TachyonRenderer, Viewport
    print(f"ovito {ovito.version_string}")

    POS = traj.pos.astype(np.float64)
    data = build_data_collection(traj, args.style, args.bg)
    pipeline = Pipeline(source=StaticSource(data=data))

    def set_positions(frame, data):
        # `frame` is the trajectory index passed to render_image(frame=k)
        data.particles_.positions_[:] = POS[frame]

    pipeline.modifiers.append(set_positions)
    pipeline.add_to_scene()

    vp = Viewport(type=Viewport.Type.Perspective)
    vp.camera_up = (0.0, 0.0, 1.0)
    vp.fov = cam.fov
    # diagnostic only: compare our orbit-wide fit with OVITO's single-frame zoom_all
    pose0 = cam.pose(0)
    vp.camera_dir = tuple(pose0["dir"])
    vp.zoom_all(size=(args.width, args.height))
    d_zoom = float(np.linalg.norm(np.asarray(vp.camera_pos) - cam.center))
    print(f"zoom_all (frame 0 only) would use distance {d_zoom:.3f}; orbit-wide fixed fit uses {cam.distance:.3f}")
    vp.fov = cam.fov

    settings = dict(TACHYON_PRESETS[(args.style, args.quality)])
    if args.ao_samples is not None:
        settings["ambient_occlusion_samples"] = args.ao_samples
    if args.aa_samples is not None:
        settings["antialiasing_samples"] = args.aa_samples
        settings["antialiasing"] = args.aa_samples > 1
    if args.ao_brightness is not None:
        settings["ambient_occlusion_brightness"] = args.ao_brightness
    if args.no_shadows:
        if args.style == "xray":
            print("WARNING: --no-shadows ignored for --style xray (Tachyon drops transparent spheres "
                  "when both AO and shadows are off)")
        else:
            settings["shadows"] = False
    renderer = TachyonRenderer(**settings)
    print("tachyon:", settings)
    bg = BG[args.bg]
    size = (args.width, args.height)

    def render_rgb(j, k):
        pose = cam.pose(j)
        vp.camera_pos = tuple(pose["pos"])
        vp.camera_dir = tuple(pose["dir"])
        vp.fov = cam.fov
        img = vp.render_image(size=size, frame=int(k), renderer=renderer, background=bg)
        return qimage_to_rgb(img)

    print(f"setup {time.time() - t_setup:.1f}s; rendering to {frames_dir}", flush=True)
    render_frames(args, traj, list(enumerate(indices)), render_rgb, frames_dir, label, tag="[ovito] ")
    finish_movie(args, frames_dir, out, len(indices))


if __name__ == "__main__":
    main()
