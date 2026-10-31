"""Small mathematical contracts; geometric success still requires real data."""
from dataclasses import replace
import numpy as np
from head_rgbd.geometry_d import (Settings, child_landmarks, physical_regularizers,
                                 safety, subdivide, area_quadrature)


def test_duplicate_depth_rays_do_not_change_normal_equations():
    A=np.array([[1.,0.],[.2,.8],[0.,1.]])
    b=np.array([1.,2.,3.]); face=np.array([0,0,1]); area=np.array([2.,1.])
    w=area_quadrature(face,area)
    repeat=np.repeat(np.arange(3),7)
    repeated=area_quadrature(face[repeat],area)
    np.testing.assert_allclose(A.T@(w[:,None]*A),A[repeat].T@(repeated[:,None]*A[repeat]))
    np.testing.assert_allclose(A.T@(w*b),A[repeat].T@(repeated*b[repeat]))


def test_subdivision_maps_landmarks_and_adds_real_topology():
    v = np.array([[0.,0.,0.], [.01,0.,0.], [0.,.01,0.]])
    f = np.array([[0,1,2]])
    ff, P, edges, parent = subdivide(f, len(v))
    bary = np.array([[.7,.2,.1], [.1,.7,.2], [.2,.1,.7], [1/3,1/3,1/3], [.5,.5,0]])
    ids, weights = child_landmarks(np.zeros(len(bary), int), bary)
    fine = P@v
    np.testing.assert_allclose((fine[ff[ids]]*weights[:,:,None]).sum(1), bary@v, atol=1e-14)
    assert fine.shape==(6,3) and ff.shape==(4,3)
    np.testing.assert_allclose(P.sum(1), 1.)


def test_physical_regularizer_is_density_independent_and_has_no_rigid_offset_gradient():
    v = np.array([[0.,0.,0.], [.01,0.,0.], [.01,.01,0.], [0.,.01,0.]])
    f = np.array([[0,1,2],[0,2,3]])
    settings = Settings()
    matrix, edges, lengths, areas, mass = physical_regularizers(v, f, settings)
    other, *_ = physical_regularizers(v, f, replace(settings, points=10000))
    np.testing.assert_array_equal(matrix.toarray(), other.toarray())
    np.testing.assert_allclose((matrix@np.ones(4))[:-4], 0, atol=1e-12)


def test_shared_safety_rejects_overbudget_and_steep_offsets():
    base = np.array([[[0.,0.,-.5], [.01,0.,-.5], [0.,.01,-.5]]])
    directions = np.zeros_like(base); directions[:,:,2]=1
    faces = np.array([[0,1,2]])
    _, edges, lengths, *_ = physical_regularizers(base[0], faces, Settings())
    assert safety(base, directions, faces, np.full(3, 1.), edges, lengths, Settings())[0]
    assert not safety(base, directions, faces, np.full(3, 3.01), edges, lengths, Settings())[0]
    assert not safety(base, directions, faces, np.array([3.,-3.,0.]), edges, lengths, Settings())[0]
