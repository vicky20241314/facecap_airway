"""Sparse constrained least squares, rather than post-hoc vertex clipping.

Implemented independently from OSQP's documented public Python interface.
"""
import numpy as np
from scipy import sparse
import osqp


def constraints(base, directions, faces, edges, lengths, settings, active_vertices=None):
    count=base.shape[1]
    if active_vertices is not None:
        # This only omits rows with no active unknown; solve() already removed
        # those rows. All faces influenced by an active vertex remain checked.
        edge_active=active_vertices[edges].any(1)
        edges,lengths=edges[edge_active],lengths[edge_active]
        faces=faces[active_vertices[faces].any(1)]
    row=np.repeat(np.arange(len(edges)),2)
    edge=sparse.coo_matrix((np.tile([1.,-1.],len(edges)),(row,edges.ravel())),shape=(len(edges),count)).tocsr()
    matrices=[sparse.eye(count,format='csr'),edge]
    limits=[np.full(count,settings.displacement_bound_mm),settings.maximum_slope*np.maximum(lengths,.05)]
    lows=[-limits[0],-limits[1]]
    from head_rgbd.normal_cone_d import normal_cones
    for matrix,lower,upper in normal_cones(base,directions,faces,settings.maximum_normal_change_deg):
        matrices.append(matrix);lows.append(lower);limits.append(upper)
    return sparse.vstack(matrices).tocsr(),np.concatenate(lows),np.concatenate(limits)


def solve(system,target,free,fixed,current,constraint_matrix,lower,upper):
    matrix=constraint_matrix[:,free].tocsc()
    matrix.eliminate_zeros()
    keep=np.asarray(matrix.getnnz(axis=1)).ravel()>0
    shift=constraint_matrix@fixed
    l=(lower-shift)[keep];u=(upper-shift)[keep]
    # Stay strictly inside the same feasible geometry bounds, so solver tolerance
    # cannot force a needless global half-step at an active edge constraint.
    l[np.isfinite(l)]+=1e-5
    u[np.isfinite(u)]-=1e-5
    matrix=matrix[keep].tocsc()
    hessian=(system.T@system).tocsc()
    linear=-np.asarray(system.T@target).ravel()
    # Scale the objective only; the feasible geometry set is unchanged.
    diagonal=hessian.diagonal()
    scale=1/max(float(np.median(diagonal[diagonal>0])),1e-12)
    solver=osqp.OSQP()
    solver.setup(P=sparse.triu(hessian*scale,format='csc'),q=linear*scale,
                 A=matrix,l=l,u=u,verbose=False,eps_abs=1e-6,eps_rel=1e-6,
                 max_iter=20000,polishing=True,adaptive_rho=True)
    solver.warm_start(x=current[free])
    result=solver.solve(raise_error=False)
    if result.info.status_val not in (1,2) or not np.isfinite(result.x).all():
        raise RuntimeError(f'constrained QP did not solve: {result.info.status}')
    return result.x,dict(status=int(result.info.status_val),status_text=result.info.status,
                         primal_residual=float(result.info.prim_res),dual_residual=float(result.info.dual_res),
                         iterations=int(result.info.iter),constraint_rows=matrix.shape[0])
