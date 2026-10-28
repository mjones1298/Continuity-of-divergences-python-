# Patched global search (P2): sqrt via SVD of the Kraus factor (exact subspace, no threshold),
# every restart's result re-verified at 50 digits (p2_mpcheck.check).
import numpy as np, sys, scipy.optimize as so
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))  # glib.py in code/common
HERE = Path(__file__).resolve().parent
from glib import *
from p2_mpcheck import best_dir, JDp, phi_kraus, check
dX=dY=2
def F_factor(Ks,Bm):
    W=np.stack([K.reshape(-1) for K in Ks],axis=1)          # J(Phi)=W W^*
    w,V=np.linalg.eigh(herm(Bm)); Bh=(V*np.sqrt(w))@V.conj().T; Bih=(V/np.sqrt(w))@V.conj().T
    U,s,_=np.linalg.svd(Bih@W,full_matrices=False)           # (Y Y^*)^{1/2} = U diag(s) U^*
    G=Bh@(U*s)@U.conj().T@Bh
    return np.linalg.eigvalsh(trY(herm(G),dY,dX)).min()
def grad(Ks,B,h):
    n=4; G=np.zeros((n,n),complex)
    for i in range(n):
        for j in range(i,n):
            Es=[]
            if i==j: E=np.zeros((n,n),complex); E[i,i]=1; Es.append(E)
            else:
                E=np.zeros((n,n),complex); E[i,j]=E[j,i]=1/np.sqrt(2); Es.append(E)
                E=np.zeros((n,n),complex); E[i,j]=1j/np.sqrt(2); E[j,i]=-1j/np.sqrt(2); Es.append(E)
            for E in Es: G+=(F_factor(Ks,B+h*E)-F_factor(Ks,B-h*E))/(2*h)*E
    return G
p=float(sys.argv[1]); nrest=int(sys.argv[2]); B=JDp(p); h=1e-6*p
rng=np.random.default_rng(int(sys.argv[3]) if len(sys.argv)>3 else 7)
obj=lambda x: -best_dir(grad(phi_kraus(x),B,h))*p
starts=[np.load(HERE / f'p2_xunit_{p:g}.npy')] if len(sys.argv)>4 else []
starts+=[rng.normal(size=32) for _ in range(nrest)]
best=0
for r,x0 in enumerate(starts):
    res=so.minimize(obj,x0,method='Nelder-Mead',options={'maxiter':900,'xatol':1e-6,'fatol':1e-7})
    np.save(HERE / f'p2v2_x_{p:g}_s{sys.argv[3]}_{r}.npy',res.x)
    print(f'p={p} restart {r}: search value p*sup = {-res.fun:.6f}',flush=True)
    check(res.x,p,f'   mp-verify restart {r}')
    best=max(best,-res.fun)
print(f'p={p}: best = {best:.6f}  (unitary closed form 0.384900; proved bound 1)',flush=True)
