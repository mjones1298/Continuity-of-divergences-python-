"""Exact worst-case first-order decrease rate of F_hat at depolarized qubit gate pairs
(N_p o U, N_p o U_delta), over trace-preserving perturbations of the FIRST channel, per unit
diamond norm.  Uses gradient M_x = S^* Z S (S=(A#B)A^{-1}, S^*Z+ZS = 1(x)xx^*) and an SDP for
max tr(E M_x) s.t. ||Delta||_dia<=1, Tr_Y E=0 (Watrous dual form). Also checks gradient by FD."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))  # shared modules in code/common (glib.py)
import numpy as np, cvxpy as cp, glib as g, scipy.linalg as sla
X=np.array([[0,1],[1,0]],complex)
def gate(p,delta):
    U=sla.expm(-0.5j*delta*X); v=g.vec(U); return (1-p)*np.outer(v,v.conj())+p/2*np.eye(4)
def grad(A,B,x):
    G=g.gmean_pd(A,B); S=G@np.linalg.inv(A)
    Y=np.kron(np.eye(2),np.outer(x,x.conj()))
    Z=sla.solve_sylvester(S.conj().T, S, Y)       # S^* Z + Z S = Y
    return g.herm(S.conj().T@Z@S)
def worst_rate(M):
    # max tr(E M) s.t. Tr_Y E = 0 and ||Delta_E||_dia <= 1, where for trace-annihilating Hermitian E
    # ||Delta_E||_dia = 2 min{ ||Tr_Y Z||_inf : Z >= 0, Z >= E }  (dual of Watrous's SDP)
    n=4; E=cp.Variable((n,n),hermitian=True); Zv=cp.Variable((n,n),hermitian=True)
    cons=[Zv>>0, Zv-E>>0, 0.5*np.eye(2)-cp.partial_trace(Zv,[2,2],axis=0)>>0, cp.partial_trace(E,[2,2],axis=0)==0]
    pr=cp.Problem(cp.Maximize(cp.real(cp.trace(E@M))),cons); pr.solve(solver='CLARABEL')
    return pr.value, E.value
rng=np.random.default_rng(0)
# FD check of gradient
A=gate(0.05,0.3); B=gate(0.05,0.0); x=np.array([1,0],complex)
E=g.herm(rng.normal(size=(4,4))+1j*rng.normal(size=(4,4))); h=1e-6
fd=(np.vdot(x,g.trY(g.gmean_pd(A+h*E,B),2,2)@x)-np.vdot(x,g.trY(g.gmean_pd(A-h*E,B),2,2)@x)).real/(2*h)
print("grad FD check:", fd, np.trace(grad(A,B,x)@E).real)
# sanity of diamond SDP: for E = J(Ad_U)-J(id) diamond = 2 sin(theta)
U=sla.expm(-0.5j*0.4*X); Ed=np.outer(g.vec(U),g.vec(U).conj())-np.outer(g.vec(np.eye(2)),g.vec(np.eye(2)))
print("diamond sanity:", g.dnorm(Ed,2,2), 2*np.sin(0.2))
val,Eo=worst_rate(np.zeros((4,4)))
# check: optimal E of a random M has diamond norm 1
Mr=g.herm(rng.normal(size=(4,4))); v,Eo=worst_rate(Mr); print("SDP check: diamond of optimiser", g.dnorm(Eo,2,2))
xs=[np.array([np.cos(a/2),np.exp(1j*b)*np.sin(a/2)]) for a in np.linspace(0,np.pi,5) for b in [0,np.pi/2]]
print(" p        delta/sqrt(p)  F_hat    worst dec. rate   rate*sqrt(p)   rate*p")
for p in [1e-2,1e-3,1e-4]:
    for y in [0.0,0.5,1.0,2.0,5.0]:
        delta=y*np.sqrt(p)*2   # x ~ delta^2/(4p) = y^2
        J1=gate(p,delta); J2=gate(p,0.0); F=g.Fhat(J1,J2,2,2)
        best=0
        for x in xs:
            val,_=worst_rate(grad(J1,J2,x)); best=max(best,val)
        print(f"{p:7.0e}  {y:5.2f}      {F:.4f}   {best:10.4f}   {best*np.sqrt(p):10.4f}   {best*p:10.5f}")
