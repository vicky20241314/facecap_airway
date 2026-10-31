"""Coupled linearized normal cones in every actual animation state.

Constraining tangent components alone ignores changes in projected area. The
cone must instead compare them to the current predicted normal component.
"""
import numpy as np
from scipy import sparse


def normal_cones(base, directions, faces, maximum_angle_deg):
    count=base.shape[1]
    coefficient=np.tan(np.radians(maximum_angle_deg*.8))/np.sqrt(2)
    for v,normals in zip(base,directions):
        tri=v[faces];n=normals[faces]
        e1,e2=tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]
        cross=np.cross(e1,e2);area=np.maximum(np.linalg.norm(cross,axis=1),1e-14)
        unit=cross/area[:,None]
        tangent=e1/np.maximum(np.linalg.norm(e1,axis=1,keepdims=True),1e-14)
        tangent2=np.cross(unit,tangent)
        jac=np.stack((-np.cross(n[:,0],e2)-np.cross(e1,n[:,0]),
                      np.cross(n[:,1],e2),np.cross(e1,n[:,2])),axis=1)*.001/area[:,None,None]
        normal_derivative=(jac*unit[:,None]).sum(2)
        def matrix(values):
            return sparse.coo_matrix((values.ravel(),(np.repeat(np.arange(len(faces)),3),faces.ravel())),
                                     shape=(len(faces),count)).tocsr()
        for axis in (tangent,tangent2):
            transverse=(jac*axis[:,None]).sum(2)
            for sign in (-1.,1.):
                # +/- tangent <= c * (1 + projected-area derivative).
                yield matrix(sign*transverse-coefficient*normal_derivative),np.full(len(faces),-np.inf),np.full(len(faces),coefficient)
        yield matrix(normal_derivative),np.full(len(faces),-.9),np.full(len(faces),np.inf)
