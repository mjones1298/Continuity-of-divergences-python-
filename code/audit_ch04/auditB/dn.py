import numpy as np, cvxpy as cp
from core import herm, ptrace_out

def diamond_herm(J, dY, dX, solver=cp.SCS, eps=1e-8):
    """Diamond norm of Hermitian-preserving map with Choi J (Y (x) X, Watrous convention).
    max <J, P0-P1> s.t. P0,P1>=0, P0+P1 <= 1_Y (x) sigma, sigma density on X."""
    n = dY*dX
    P0 = cp.Variable((n,n), hermitian=True); P1 = cp.Variable((n,n), hermitian=True)
    s = cp.Variable((dX,dX), hermitian=True)
    cons = [P0>>0, P1>>0, s>>0, cp.real(cp.trace(s))==1,
            cp.kron(np.eye(dY), s) - P0 - P1 >> 0]
    obj = cp.Maximize(cp.real(cp.trace(J @ (P0-P1))))
    prob = cp.Problem(obj, cons)
    prob.solve(solver=solver, eps=eps, max_iters=200000)
    return prob.value

if __name__=="__main__":
    # sanity: identity channel diamond norm 1, difference of two replacement channels
    dX,dY=2,2
    y1=np.array([1,0.]);y2=np.array([np.cos(0.3),np.sin(0.3)])
    phi=np.outer(y1,y1);psi=np.outer(y2,y2)
    J=np.kron(phi-psi,np.eye(dX))
    print(diamond_herm(J,dY,dX), 2*np.sin(0.3))
