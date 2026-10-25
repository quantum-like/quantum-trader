"""Trajectory I/O following FORMAT.md (fbdmd-hs-movies).

Readers:  load_traj(path) -> Trajectory   (.npz cache preferred, extended-XYZ fallback)
Writers:  write_npz / write_xyz / write_json (used by make_test_traj.py; the real
          simulator has its own writer but produces the same layout).

Only numpy is required.
"""
from __future__ import annotations

import json
import os
import re

import numpy as np

TYPE_TO_SPECIES = {1: "HS", 2: "POS", 3: "NEG"}
SPECIES_TO_TYPE = {"HS": 1, "POS": 2, "NEG": 3}
TYPE_TO_CHARGE = {1: 0, 2: 1, 3: -1}


class Trajectory:
    """In-memory trajectory: pos (F,N,3) float32 wrapped into [0,L), types (N,) int8 1/2/3,
    charge (N,) int8, times (F,) float64, L float, meta dict."""

    def __init__(self, pos, types, charge, times, L, meta=None, source=None):
        self.pos = np.ascontiguousarray(pos, dtype=np.float32)
        self.types = np.asarray(types, dtype=np.int8)
        self.charge = np.asarray(charge, dtype=np.int8)
        self.times = np.asarray(times, dtype=np.float64)
        self.L = float(L)
        self.meta = dict(meta or {})
        self.source = source
        if self.pos.ndim != 3 or self.pos.shape[2] != 3:
            raise ValueError(f"pos must be (F,N,3), got {self.pos.shape}")
        if self.types.shape != (self.pos.shape[1],):
            raise ValueError("types must have shape (N,)")
        if self.times.shape != (self.pos.shape[0],):
            raise ValueError("times must have shape (F,)")

    @property
    def nframes(self):
        return self.pos.shape[0]

    @property
    def N(self):
        return self.pos.shape[1]

    def counts(self):
        return {t: int((self.types == t).sum()) for t in (1, 2, 3)}

    def describe(self):
        c = self.counts()
        return (f"{self.nframes} frames, N={self.N} (HS {c[1]}, + {c[2]}, - {c[3]}), "
                f"L={self.L:.4f}, t={self.times[0]:.3f}..{self.times[-1]:.3f}")


# ----------------------------------------------------------------------------- readers

def load_traj(path: str) -> Trajectory:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npz":
        return load_npz(path)
    if ext in (".xyz", ".exyz", ".extxyz"):
        return load_xyz(path)
    raise ValueError(f"unknown trajectory format: {path} (expected .npz or .xyz)")


def _sidecar_meta(path):
    jpath = os.path.splitext(path)[0] + ".json"
    if os.path.exists(jpath):
        try:
            with open(jpath) as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def load_npz(path: str) -> Trajectory:
    with np.load(path, allow_pickle=False) as z:
        pos = z["pos"]
        types = z["types"]
        charge = z["charge"] if "charge" in z.files else np.vectorize(TYPE_TO_CHARGE.get)(types)
        times = z["times"] if "times" in z.files else np.arange(pos.shape[0], dtype=float)
        L = float(np.asarray(z["L"]).reshape(-1)[0])
        meta = {}
        if "meta" in z.files:
            try:
                meta = json.loads(str(np.asarray(z["meta"]).reshape(-1)[0]))
            except Exception:
                meta = {}
    if not meta:
        meta = _sidecar_meta(path)
    return Trajectory(pos, types, charge, times, L, meta, source=path)


_PROP_RE = re.compile(r"Properties=(\S+)")
_LAT_RE = re.compile(r'Lattice="([^"]+)"')
_TIME_RE = re.compile(r"Time=([-+0-9.eE]+)")


def _parse_properties(spec: str):
    """'species:S:1:pos:R:3:id:I:1' -> {name: (col0, ncols)}"""
    toks = spec.split(":")
    cols = {}
    c = 0
    for i in range(0, len(toks) - 2, 3):
        name, _kind, n = toks[i], toks[i + 1], int(toks[i + 2])
        cols[name] = (c, n)
        c += n
    return cols


def load_xyz(path: str) -> Trajectory:
    with open(path) as f:
        lines = f.read().split("\n")
    frames, times = [], []
    L = None
    types = charge = None
    i, nl = 0, len(lines)
    while i < nl:
        s = lines[i].strip()
        if not s:
            i += 1
            continue
        n = int(s)
        comment = lines[i + 1] if i + 1 < nl else ""
        block = lines[i + 2:i + 2 + n]
        if len(block) < n:
            break  # truncated last frame: ignore
        m = _LAT_RE.search(comment)
        if m and L is None:
            vals = [float(v) for v in m.group(1).split()]
            L = vals[0]
        m = _TIME_RE.search(comment)
        times.append(float(m.group(1)) if m else float(len(frames)))
        m = _PROP_RE.search(comment)
        cols = _parse_properties(m.group(1)) if m else {"species": (0, 1), "pos": (1, 3)}
        arr = np.array([l.split() for l in block])
        p0 = cols["pos"][0]
        frames.append(arr[:, p0:p0 + 3].astype(np.float32))
        if types is None:
            if "type" in cols:
                types = arr[:, cols["type"][0]].astype(int)
            elif "species" in cols:
                types = np.array([SPECIES_TO_TYPE.get(sp, 1) for sp in arr[:, cols["species"][0]]])
            else:
                types = np.ones(n, int)
            if "charge" in cols:
                charge = np.rint(arr[:, cols["charge"][0]].astype(float)).astype(int)
            else:
                charge = np.vectorize(TYPE_TO_CHARGE.get)(types)
        i += 2 + n
    if not frames:
        raise ValueError(f"no frames found in {path}")
    pos = np.stack(frames)
    if L is None:
        L = float(np.ceil(pos.max() + 0.5))
    meta = _sidecar_meta(path)
    return Trajectory(pos, types, charge, np.array(times), L, meta, source=path)


# ----------------------------------------------------------------------------- writers

def write_npz(path, pos, types, charge, times, L, meta):
    np.savez(path,
             pos=np.asarray(pos, np.float32),
             types=np.asarray(types, np.int8),
             charge=np.asarray(charge, np.int8),
             times=np.asarray(times, np.float64),
             L=np.float64(L),
             meta=json.dumps(meta))


def write_json(path, meta):
    with open(path, "w") as f:
        json.dump(meta, f, indent=1)
        f.write("\n")


def write_xyz(path, pos, types, charge, times, L):
    pos = np.asarray(pos)
    F, N = pos.shape[:2]
    species = np.array(["HS", "POS", "NEG"])[np.asarray(types, int) - 1]
    ids = np.arange(1, N + 1)
    header = ('Lattice="{L:.6f} 0 0 0 {L:.6f} 0 0 0 {L:.6f}" '
              'Properties=species:S:1:pos:R:3:id:I:1:type:I:1:charge:R:1 '
              'Time={t:.6f} Frame={k} pbc="T T T"\n')
    with open(path, "w") as f:
        for k in range(F):
            f.write(f"{N}\n")
            f.write(header.format(L=L, t=float(times[k]), k=k))
            f.write("\n".join(
                f"{s} {x:.6f} {y:.6f} {z:.6f} {i} {t} {c}"
                for s, (x, y, z), i, t, c in zip(species, pos[k], ids, types, charge)))
            f.write("\n")
