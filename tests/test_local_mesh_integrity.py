"""Geometric invariants, including the two-split-edge green-refinement regression."""
import itertools

import numpy as np
import pytest

from head_rgbd.mesh_integrity import orient_faces_consistently, topology_report
from head_rgbd.neck_subject_v3 import subdivide_local


@pytest.mark.parametrize("mask", list(itertools.product((False, True), repeat=4)))
def test_local_refinement_preserves_parent_orientation_and_closed_surface(mask):
    vertices = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
    faces = np.array([[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]])
    refined = subdivide_local(vertices, faces, face_mask=np.array(mask))
    old = vertices[faces]
    new = refined["vertices"][refined["faces"]]
    old_normals = np.cross(old[:, 1] - old[:, 0], old[:, 2] - old[:, 0])
    new_normals = np.cross(new[:, 1] - new[:, 0], new[:, 2] - new[:, 0])
    assert np.all(np.sum(new_normals * old_normals[refined["parent_faces"]], axis=1) > 0)
    child_areas = np.bincount(refined["parent_faces"], weights=np.linalg.norm(new_normals, axis=1))
    np.testing.assert_allclose(child_areas, np.linalg.norm(old_normals, axis=1))
    audit = topology_report(refined["vertices"], refined["faces"])
    assert audit["inconsistent_shared_edges"] == 0
    assert audit["nonmanifold_edges"] == 0
    assert audit["boundary_edges"] == 0


def test_orientation_repair_preserves_vertices_and_boundary():
    vertices = np.array([[0., 0., 0.], [1., 0., 0.], [1., 1., 0.], [0., 1., 0.]])
    faces = np.array([[0, 1, 2], [0, 3, 2]])
    assert topology_report(vertices, faces)["inconsistent_shared_edges"] == 1
    repaired, record = orient_faces_consistently(faces)
    assert record["reversed_faces"] == 1
    assert topology_report(vertices, repaired)["inconsistent_shared_edges"] == 0
    assert topology_report(vertices, repaired)["boundary_edges"] == 4
