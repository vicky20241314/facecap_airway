"""Camera-origin visibility queries with an exact narrow phase.

Every visibility query in this pipeline casts from the camera centre, so a ray
is fully described by the surface point it was backprojected to. Projecting the
triangles onto the plane the camera looks along (``u = x/-z``, ``v = y/-z``)
turns visibility into a bounded 2D lookup: a ray can only hit a triangle whose
projected extent contains its own ``(u, v)``. The lookup table is a uniform grid
and the narrow phase is an exact Moller-Trumbore test, so first hit, ray
numbering, two-sided queries, occlusion and missing hits keep the semantics of
the per-ray rtree implementation this replaces.
"""
import numpy as np

CELLS = 160


class RayIndex:
    """Visibility accelerator for one vertex/face state.

    The index is bound to exactly the vertices it was built from; a moved surface
    needs a new one. Callers own that lifetime and build it once per state.
    """

    def __init__(self, vertices, faces, cells=CELLS):
        tri = vertices[faces]
        reach = -tri[:, :, 2]
        if not (reach > 0).all():
            raise ValueError('ray index requires every triangle in front of the camera')
        self.cells = cells
        self.tri = tri
        u = tri[:, :, 0] / reach
        v = tri[:, :, 1] / reach
        self.u0, self.u1 = float(u.min()), float(u.max())
        self.v0, self.v1 = float(v.min()), float(v.max())
        self.su = cells / (self.u1 - self.u0)
        self.sv = cells / (self.v1 - self.v0)
        iu0 = self._axis(u.min(1), self.u0, self.su)
        iu1 = self._axis(u.max(1), self.u0, self.su)
        iv0 = self._axis(v.min(1), self.v0, self.sv)
        iv1 = self._axis(v.max(1), self.v0, self.sv)
        wide = iu1 - iu0 + 1
        tall = iv1 - iv0 + 1
        per_triangle = wide * tall
        triangle = np.repeat(np.arange(len(faces)), per_triangle)
        within = np.arange(int(per_triangle.sum())) - np.repeat(np.cumsum(per_triangle) - per_triangle, per_triangle)
        tall_per_entry = np.repeat(tall, per_triangle)
        column = np.repeat(iu0, per_triangle) + within // tall_per_entry
        row = np.repeat(iv0, per_triangle) + within % tall_per_entry
        cell = column * cells + row
        order = np.argsort(cell, kind='stable')
        self.triangles = triangle[order].astype(np.int32)
        counts = np.bincount(cell, minlength=cells * cells)
        self.indptr = np.concatenate(([0], np.cumsum(counts))).astype(np.int64)

    def _axis(self, value, origin, scale):
        return np.clip((value - origin) * scale, 0, self.cells - 1).astype(np.int64)

    def first_hits(self, points, chunk=16384):
        """Closest triangle for each ray from the camera centre, in input order.

        Returns ``(hit, ray, face)`` with the same contract as the rtree path:
        ``hit`` is ``(n,3)`` metric intersection points, ``ray`` indexes into
        ``points`` and keeps the caller's numbering, ``face`` indexes ``faces``.
        A ray with no intersection is simply absent from all three arrays.
        """
        count = len(points)
        hit_ray = np.full(count, -1, np.int64)
        hit_face = np.full(count, -1, np.int64)
        hit_t = np.full(count, np.inf)
        for start in range(0, count, chunk):
            block = points[start:start + chunk]
            reach = -block[:, 2]
            with np.errstate(divide='ignore', invalid='ignore'):
                u = block[:, 0] / reach
                v = block[:, 1] / reach
            column = np.floor((u - self.u0) * self.su)
            row = np.floor((v - self.v0) * self.sv)
            inside = ((reach > 0) & (column >= 0) & (column < self.cells)
                      & (row >= 0) & (row < self.cells))
            cell = np.where(inside, column * self.cells + row, 0).astype(np.int64)
            begin = self.indptr[cell]
            span = np.where(inside, self.indptr[cell + 1] - begin, 0)
            total = int(span.sum())
            if not total:
                continue
            ray = np.repeat(np.arange(len(block)), span)
            offset = np.arange(total) - np.repeat(np.cumsum(span) - span, span)
            triangle = self.triangles[np.repeat(begin, span) + offset]
            direction = block[ray]
            tri = self.tri[triangle]
            edge1 = tri[:, 1] - tri[:, 0]
            edge2 = tri[:, 2] - tri[:, 0]
            normal = np.cross(direction, edge2)
            determinant = (edge1 * normal).sum(1)
            usable = np.abs(determinant) > 1e-12
            inverse = np.zeros_like(determinant)
            np.divide(1., determinant, out=inverse, where=usable)
            to_origin = -tri[:, 0]
            bary_u = (to_origin * normal).sum(1) * inverse
            cross = np.cross(to_origin, edge1)
            bary_v = (cross * direction).sum(1) * inverse
            distance = (edge2 * cross).sum(1) * inverse
            crossed = (usable & (bary_u >= 0.) & (bary_u <= 1.)
                       & (bary_v >= 0.) & (bary_u + bary_v <= 1.) & (distance > 0.))
            keep = np.flatnonzero(crossed)
            if not len(keep):
                continue
            local_ray = ray[keep]
            order = np.lexsort((distance[keep], local_ray))
            first = np.concatenate(([True], local_ray[order][1:] != local_ray[order][:-1]))
            best = keep[order[first]]
            hit_ray[start + ray[best]] = ray[best]
            hit_face[start + ray[best]] = triangle[best]
            hit_t[start + ray[best]] = distance[best]
        found = np.flatnonzero(hit_face >= 0)
        if not len(found):
            return np.empty((0, 3)), np.empty(0, np.int64), np.empty(0, np.int64)
        return hit_t[found, None] * points[found], found, hit_face[found]
