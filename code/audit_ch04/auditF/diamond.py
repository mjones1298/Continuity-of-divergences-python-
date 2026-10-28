import numpy as np, cvxpy as cp
def diamond_hp(J, dY, dX, solver='SCS'):
    """cb trace norm of Hermiticity-preserving map with Choi matrix J (Y(x)X). SDP: max <J, P0-P1>, P0,P1>=0, P0+P1 <= 1_Y (x) rho."""
    n = dY*dX
    P0 = cp.Variable((n, n), hermitian=True); P1 = cp.Variable((n, n), hermitian=True)
    rho = cp.Variable((dX, dX), hermitian=True)
    cons = [P0 >> 0, P1 >> 0, rho >> 0, cp.real(cp.trace(rho)) == 1, cp.kron(np.eye(dY), rho) - P0 - P1 >> 0]
    prob = cp.Problem(cp.Maximize(cp.real(cp.trace(J.conj().T @ (P0 - P1)))), cons)
    prob.solve(solver=cp.SCS, eps=1e-8, max_iters=200000)
    return prob.value
if __name__ == "__main__":
    # sanity: diamond norm of difference of two unitary qubit channels vs known formula
    import sys; sys.path.insert(0, '.')
    from lib import vec
    U = np.eye(2, dtype=complex); V = np.diag([1, np.exp(1j*0.7)])
    JU = np.outer(vec(U), vec(U).conj()); JV = np.outer(vec(V), vec(V).conj())
    d = diamond_hp(JU - JV, 2, 2)
    # diamond distance between unitary channels = 2 sqrt(1 - min_{rho} |tr(rho U^*V)|^2) ; here eigenphases 0 and 0.7 -> 2 sin(0.35)
    print(d, 2*np.sin(0.35))
