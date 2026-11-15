"""Minimal orthographic software rasteriser with a z-buffer and Gouraud shading.

The acceptance comparison must show *geometry*, not the renderer.  Painter's
algorithm with one constant colour per triangle turns a 1.7 mm mesh into a
mosaic of flat quads, which reads as faceting even when the surface is smooth --
and hides real faceting when the mesh is coarse.  This module instead
interpolates a per-vertex value barycentrically and resolves visibility with a
true depth buffer, so adjacent triangles share their shading at shared vertices
and only a genuine normal discontinuity shows up as a crease.

Orthographic projection, because the comparison is about millimetres of surface
relief and a perspective divide would make the panels incomparable.
"""
from __future__ import annotations

import numpy as np

from head_rgbd import neck_prominence as npn

WORLD_UP = np.array([0.0, 1.0, 0.0])


def basis(view) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Camera basis for a view direction ``fwd`` pointing from the eye into the scene.

    ``right = fwd x up`` keeps the frame right handed, so screen +x is world +x
    on the front view and the image is not mirrored.  In this project +x is the
    subject's left (a negative yaw brings +x to the camera), therefore a front
    view puts the subject's left on the right of the image, as when looking at a
    person facing you.
    """
    fwd = np.asarray(view, float)
    fwd = fwd / np.linalg.norm(fwd)
    right = np.cross(fwd, WORLD_UP)
    n = np.linalg.norm(right)
    if n < 1e-9:                      # looking straight down/up
        right = np.array([1.0, 0.0, 0.0])
    else:
        right = right / n
    up = np.cross(right, fwd)
    up /= np.linalg.norm(up)
    return fwd, right, up


def light_vector(view, azimuth_deg=0.0, elevation_deg=22.0):
    """Light direction in world coordinates for a view.

    ``azimuth`` rotates the light about the camera axis (0 = from the eye side,
    positive = towards screen right); ``elevation`` lifts it above the camera
    plane.  A low elevation (5-12 deg) grazes the surface and is what makes a
    1-3 mm protrusion legible.
    """
    fwd, right, up = basis(view)
    az = np.radians(azimuth_deg)
    el = np.radians(elevation_deg)
    d = (-fwd) * np.cos(el) * np.cos(az) + right * np.cos(el) * np.sin(az) + up * np.sin(el)
    return d / np.linalg.norm(d)


def _rasterise(xy, z, faces, values, size, zbuf, sbuf):
    """Fill one image from pre-projected triangles; depth decides visibility."""
    ax, ay = xy[:, 0], xy[:, 1]
    C = values.shape[1]
    for a, b, c in faces:
        x0, x1, x2 = ax[a], ax[b], ax[c]
        y0, y1, y2 = ay[a], ay[b], ay[c]
        xlo = int(np.floor(min(x0, x1, x2)))
        xhi = int(np.ceil(max(x0, x1, x2)))
        ylo = int(np.floor(min(y0, y1, y2)))
        yhi = int(np.ceil(max(y0, y1, y2)))
        if xhi < 0 or yhi < 0 or xlo > size - 1 or ylo > size - 1:
            continue
        xlo = max(xlo, 0)
        ylo = max(ylo, 0)
        xhi = min(xhi, size - 1)
        yhi = min(yhi, size - 1)
        den = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
        if abs(den) < 1e-12:
            continue
        gx, gy = np.meshgrid(np.arange(xlo, xhi + 1) + 0.5,
                             np.arange(ylo, yhi + 1) + 0.5)
        l0 = ((y1 - y2) * (gx - x2) + (x2 - x1) * (gy - y2)) / den
        l1 = ((y2 - y0) * (gx - x2) + (x0 - x2) * (gy - y2)) / den
        l2 = 1.0 - l0 - l1
        inside = (l0 >= -1e-7) & (l1 >= -1e-7) & (l2 >= -1e-7)
        if not inside.any():
            continue
        zz = l0 * z[a] + l1 * z[b] + l2 * z[c]
        sub = zbuf[ylo:yhi + 1, xlo:xhi + 1]
        take = inside & (zz < sub)
        if not take.any():
            continue
        sub[take] = zz[take]
        ssub = sbuf[ylo:yhi + 1, xlo:xhi + 1]
        for ch in range(C):
            ss = l0 * values[a, ch] + l1 * values[b, ch] + l2 * values[c, ch]
            target = ssub[..., ch] if C > 1 else ssub
            target[take] = ss[take]


def _project(V, faces, view, centre, half, size, keep_backface=True):
    fwd, right, up = basis(view)
    P = V - np.asarray(centre, float)
    X, Y, Z = P @ right, P @ up, P @ fwd
    scale = (0.5 * size) / half
    xy = np.stack([X * scale + 0.5 * size, 0.5 * size - Y * scale], axis=1)
    F = faces
    if keep_backface:
        tri = V[faces]
        fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        keep = (fn @ (-fwd)) > 0.0
        if keep.any():
            F = faces[keep]
    return xy, Z, F, fwd


MM_PER_UNIT = 1000.0        # the OBJ files of this project are in metres


def render(vertices, faces, view, centre, half, px=520, ss=2, light=None,
           vn=None, ambient=0.18, diffuse=0.82, colour=(0.97, 0.95, 0.92),
           bg=(0.105, 0.105, 0.115), unit_scale=MM_PER_UNIT):
    """Orthographic render.  Returns an ``(px, px, 3)`` float image in [0, 1].

    ``centre`` is the world point placed at the image centre, ``half`` the half
    width of the framed window in mm (so one image pixel is ``2*half/(px*ss)`` mm).
    Vertices are taken in project units (metres) and converted to mm.
    """
    V = np.asarray(vertices, float) * unit_scale
    F = np.asarray(faces, np.int64)
    centre = np.asarray(centre, float) * unit_scale
    if light is None:
        light = light_vector(view)
    if vn is None:
        vn = npn.vertex_normals(V, F)
    val = (ambient + diffuse * np.clip(vn @ light, 0.0, 1.0)).astype(np.float32)

    size = int(px * ss)
    xy, Z, Fr, _ = _project(V, F, view, centre, half, size)
    zbuf = np.full((size, size), np.inf, np.float32)
    sbuf = np.zeros((size, size), np.float32)
    _rasterise(xy, Z, Fr, val[:, None], size, zbuf, sbuf)

    filled = np.isfinite(zbuf)
    img = np.empty((size, size, 3), np.float32)
    img[:] = np.asarray(bg, np.float32)
    tone = sbuf[..., None] * np.asarray(colour, np.float32)
    img[filled] = tone[filled]
    if ss > 1:
        img = img.reshape(px, ss, px, ss, 3).mean(axis=(1, 3))
    return np.clip(img, 0.0, 1.0)


def mean_curvature(vertices, faces, vn=None):
    """Signed cotangent mean curvature along the outward normal (1/mm).

    Positive = convex (a ridge or a protrusion), negative = concave (a groove).
    """
    V = np.asarray(vertices, float)
    F = np.asarray(faces, np.int64)
    if vn is None:
        vn = npn.vertex_normals(V, F)
    i0, i1, i2 = F[:, 0], F[:, 1], F[:, 2]
    area = np.zeros(len(V), float)

    def _cot(p, q, r):
        """cot of the angle at ``r`` in the triangle (p, q, r)."""
        u, v = p - r, q - r
        cr = np.linalg.norm(np.cross(u, v), axis=1)
        dt = np.einsum("ij,ij->i", u, v)
        return dt / np.maximum(cr, 1e-12)

    tri = V[F]
    area_f = 0.5 * np.linalg.norm(
        np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    np.add.at(area, i0, area_f / 3.0)
    np.add.at(area, i1, area_f / 3.0)
    np.add.at(area, i2, area_f / 3.0)

    L = np.zeros_like(V)
    for a, b, c in ((i1, i2, i0), (i0, i2, i1), (i0, i1, i2)):
        cot = _cot(V[a], V[b], V[c])
        np.add.at(L, a, (0.5 * cot)[:, None] * (V[a] - V[b]))
        np.add.at(L, b, (0.5 * cot)[:, None] * (V[b] - V[a]))
    H = np.einsum("ij,ij->i", L, vn) / np.maximum(2.0 * area, 1e-9)
    return H


def neighbour_average(vertices, faces, values, iterations=12):
    """Average a per-vertex scalar over the 1-ring, repeated (a cheap low pass)."""
    V = np.asarray(vertices, float)
    F = np.asarray(faces, np.int64)
    e = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]], axis=0)
    e = np.sort(e, axis=1)
    e = np.unique(e, axis=0)
    a, b = e[:, 0], e[:, 1]
    x = np.asarray(values, float).copy()
    deg = np.zeros(len(V), float)
    np.add.at(deg, a, 1.0)
    np.add.at(deg, b, 1.0)
    deg = np.maximum(deg, 1.0)
    if x.ndim > 1:
        deg = deg[:, None]
    for _ in range(int(iterations)):
        acc = np.zeros_like(x)
        np.add.at(acc, a, x[b])
        np.add.at(acc, b, x[a])
        x = 0.5 * x + 0.5 * acc / deg
    return x


def render_relief(vertices, faces, view, centre, half, px=520, ss=2,
                  blur_mm=26.0, clip_pct=97.0, vn=None,
                  bg=(0.105, 0.105, 0.115)):
    """Curvature-anomaly render: red = locally convex, blue = locally concave.

    The broad cylindrical curvature of a neck is removed with a 1-ring low pass
    of roughly ``blur_mm`` radius, and the remainder is normalised by its own
    robust scale, so the colour shows *local* relief only -- a protrusion, a
    groove, a crease -- independently of the global shape or the mesh density.
    """
    V = np.asarray(vertices, float) * MM_PER_UNIT
    F = np.asarray(faces, np.int64)
    centre = np.asarray(centre, float) * MM_PER_UNIT
    if vn is None:
        vn = npn.vertex_normals(V, F)
    H = mean_curvature(V, F, vn)

    tri = V[F]
    edge = np.linalg.norm(tri[:, 1] - tri[:, 0], axis=1)
    edge = float(np.median(edge)) or 1.0
    iters = max(1, int(round(blur_mm / edge)))
    k = H - neighbour_average(V, F, H, iterations=iters)
    scale = float(np.percentile(np.abs(k), clip_pct))
    tone = np.clip(k / max(scale, 1e-9), -1.0, 1.0)
    rgb = np.stack([0.5 + 0.5 * tone,
                    0.5 - 0.30 * np.abs(tone),
                    0.5 - 0.5 * tone], axis=1).astype(np.float32)

    size = int(px * ss)
    xy, Z, Fr, _ = _project(V, F, view, centre, half, size)
    zbuf = np.full((size, size), np.inf, np.float32)
    sbuf = np.zeros((size, size, 3), np.float32)
    _rasterise(xy, Z, Fr, rgb, size, zbuf, sbuf)
    filled = np.isfinite(zbuf)
    out = np.empty((size, size, 3), np.float32)
    out[:] = np.asarray(bg, np.float32)
    out[filled] = sbuf[filled]
    if ss > 1:
        out = out.reshape(px, ss, px, ss, 3).mean(axis=(1, 3))
    return np.clip(out, 0.0, 1.0)
