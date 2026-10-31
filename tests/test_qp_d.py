import numpy as np
from scipy import sparse
from head_rgbd.qp_d import solve


def test_explicit_slope_constraint_does_not_shrink_an_unrelated_identity_region():
    # Two adjacent vertices want an impossible jump; a third disconnected region
    # should still recover its supported displacement, unlike one global scaling.
    system=sparse.eye(3,format='csr');target=np.array([2.,-2.,1.5])
    constraint=sparse.vstack((sparse.eye(3),sparse.csr_matrix([[1.,-1.,0.]]))).tocsr()
    lower=np.array([-3.,-3.,-3.,-.5]);upper=-lower
    x,info=solve(system,target,np.arange(3),np.zeros(3),np.zeros(3),constraint,lower,upper)
    assert info['status'] in (1,2)
    assert abs(x[0]-x[1])<=.50001
    assert abs(x[2]-1.5)<1e-5
