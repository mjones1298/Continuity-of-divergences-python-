# Lemma J: ||J(Delta)||_inf <= (d_X/2)||Delta||_dia for trace-annihilating Hermitian-preserving Delta; <= d_X||Delta||_dia in general.
import numpy as np, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))  # glib.py in code/common
from glib import *
rng=np.random.default_rng(1)
def maxent(d):
    v=np.eye(d).reshape(-1); return np.outer(v,v)
def dnorm_general(Jd,dY,dX):
    # diamond norm of a Hermitian-preserving map (Watrous SDP, general form)
    import cvxpy as cp
    n=dY*dX
    Y0=cp.Variable((n,n),hermitian=True); Y1=cp.Variable((n,n),hermitian=True)
    r0=cp.Variable((dX,dX),hermitian=True); r1=cp.Variable((dX,dX),hermitian=True)
    # primal: max <J,X> ... use simple dual: ||Phi||_dia = min (||Tr_Y Y0|| + ||Tr_Y Y1||)/2 s.t. [[Y0,-J],[-J,Y1]]>=0
    M=cp.bmat([[Y0,-Jd],[-Jd.conj().T,Y1]])
    t0=cp.Variable(); t1=cp.Variable()
    def trY_cp(Y):
        return sum(Y[np.ix_([i*dX+a for a in range(dX)],[i*dX+b for b in range(dX)])] for i in range(dY))
    cons=[M>>0, trY_cp(Y0)<<t0*np.eye(dX), trY_cp(Y1)<<t1*np.eye(dX)]
    pr=cp.Problem(cp.Minimize((t0+t1)/2),cons); pr.solve(solver='CLARABEL')
    return pr.value
worst=0
for d in [2,3]:
    for trial in range(30 if d==2 else 15):
        K1=rand_kraus(rng,d,d,rng.integers(1,d*d+1)); K2=rand_kraus(rng,d,d,rng.integers(1,d*d+1))
        if trial%3==0:  # near-unitary pairs
            U=rand_unitary(rng,d); V=U@fn(herm(rng.normal(size=(d,d))+0j),lambda w: np.exp(0.0*w))
            K1=[U]; K2=[rand_unitary(rng,d)]
        Jd=choi(K1)-choi(K2)
        r=np.abs(np.linalg.eigvalsh(Jd)).max()/dnorm_general(Jd,d,d)
        worst=max(worst,r/(d/2))
    print(f'd={d}: max ||J||/((d/2)||.||_dia) so far = {worst:.6f}')
for d in [2,3,4]:
    Jid=maxent(d); Jdep=np.eye(d*d)/d
    Jd=Jid-Jdep
    print(f'd={d}: id-Dep: ||J||={np.abs(np.linalg.eigvalsh(Jd)).max():.6f}, (d/2)||.||_dia={d/2*dnorm_general(Jd,d,d):.6f}, analytic d-1/d={d-1/d:.6f}')
    print(f'      identity map: ||J||/||.||_dia = {np.abs(np.linalg.eigvalsh(Jid)).max()/dnorm_general(Jid,d,d):.6f} (=d)')
