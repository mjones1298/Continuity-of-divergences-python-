# Helpers for the P2 slope computations (Prop. L2): the part of the original p2_worst.py before
# unitary_exact(), unchanged. Executed by p2_unitary.py, p2_mpcheck.py and p2_resolve.py.
# P2: sup over Phi and TP unit-diamond directions E of dF/de at Psi = D_p (qubit), F=Fhat(Phi,Psi+eE)
import numpy as np, sys, cvxpy as cp, scipy.optimize as so
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))  # glib.py in code/common
from glib import *
dX=dY=2
X=np.array([[0,1],[1,0]],complex)
def JDp(p): return (1-p)*choi([np.eye(2)])+p/2*np.eye(4)
def trYcp(Y): return sum(Y[np.ix_([i*dX+a for a in range(dX)],[i*dX+b for b in range(dX)])] for i in range(dY))
def grad_op(JA,B,h=1e-7):
    # G with dF[E] = Re Tr(E G) (finite differences on hermitian basis), F = lam_min TrY(B#A)
    n=4; basis=[]
    for i in range(n):
        for j in range(i,n):
            if i==j: E=np.zeros((n,n),complex); E[i,i]=1; basis.append(E)
            else:
                E=np.zeros((n,n),complex); E[i,j]=E[j,i]=1/np.sqrt(2); basis.append(E)
                E=np.zeros((n,n),complex); E[i,j]=1j/np.sqrt(2); E[j,i]=-1j/np.sqrt(2); basis.append(E)
    def gm(Bm,A):
        Bh=sqrtm(Bm); Bih=fn(Bm,lambda w: w**-0.5); M=herm(Bih@A@Bih); w,U=np.linalg.eigh(M)
        w=np.where(w>1e-11*max(1,w.max()),w,0.0); return herm(Bh@(U*np.sqrt(w))@U.conj().T@Bh)
    f=lambda Bm: np.linalg.eigvalsh(trY(gm(Bm,JA),dY,dX)).min()
    G=np.zeros((n,n),complex)
    for E in basis:
        g=(f(B+h*E)-f(B-h*E))/(2*h); G+=g*E
    return G
def best_dir(G):
    n=4; E=cp.Variable((n,n),hermitian=True); Y0=cp.Variable((n,n),hermitian=True); Y1=cp.Variable((n,n),hermitian=True)
    t0=cp.Variable(); t1=cp.Variable()
    cons=[cp.bmat([[Y0,-E],[-E,Y1]])>>0, trYcp(Y0)<<t0*np.eye(2), trYcp(Y1)<<t1*np.eye(2), (t0+t1)/2<=1, trYcp(E)==0]
    pr=cp.Problem(cp.Maximize(cp.real(cp.trace(E@G))),cons); pr.solve(solver='CLARABEL')
    return pr.value
