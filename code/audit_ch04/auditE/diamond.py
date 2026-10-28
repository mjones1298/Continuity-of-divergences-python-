import numpy as np, cvxpy as cp
from lib import *

def ptrace_Y_cvx(Y, dY, dX):
    # partial trace over first factor of a (dY*dX)x(dY*dX) cvxpy expression
    return cp.partial_trace(Y, [dY, dX], axis=0)

def diamond(J, dY, dX):
    """Diamond norm of Hermitian-preserving map with Choi J (Y first), via Watrous dual SDP."""
    N = dY*dX
    J = herm(J)
    Y0 = cp.Variable((N, N), hermitian=True); Y1 = cp.Variable((N, N), hermitian=True)
    M = cp.bmat([[Y0, -J], [-J.conj().T, Y1]])
    cons = [M >> 0]
    obj = 0.5*cp.lambda_max(ptrace_Y_cvx(Y0, dY, dX)) + 0.5*cp.lambda_max(ptrace_Y_cvx(Y1, dY, dX))
    prob = cp.Problem(cp.Minimize(obj), cons)
    prob.solve(solver=cp.CLARABEL)
    return prob.value

if __name__ == '__main__':
    rng = np.random.default_rng(0)
    dY, dX = 2, 2; N = 4
    # CP map: diamond = ||Tr_Y J||
    J = rand_psd(N, 3, rng)
    print(diamond(J, dY, dX), opn(ptrace_Y(J, dY, dX)))
    # identity-vs-state replace maps
    Jd = rand_psd(N, 2, rng) - rand_psd(N, 2, rng)
    d = diamond(Jd, dY, dX)
    # lower bound by random pure inputs
    lb = 0
    for _ in range(20000):
        u = rng.standard_normal(dX*dX) + 1j*rng.standard_normal(dX*dX); u /= np.linalg.norm(u)
        A = u.reshape(dX, dX)
        # (Phi (x) 1)(uu*) = (1 (x) A^T) J (1 (x) A^T)^*
        Cm = np.kron(np.eye(dY), A.T)
        M = Cm @ Jd @ Cm.conj().T
        lb = max(lb, np.abs(np.linalg.eigvalsh(herm(M))).sum())
    print(d, lb)
