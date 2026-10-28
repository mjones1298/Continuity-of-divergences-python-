import numpy as np, sys
sys.path.insert(0,'.')
from lib import *
import importlib.util
src=open('appendix_final.py').read().split("ps=[1,1.5")[0]
exec(src)
rng=np.random.default_rng(77); worst=0; worstcb=0
for trial in range(150):
    dY=int(rng.integers(1,4)); dX=int(rng.integers(1,4)); d=dY*dX
    r1=int(rng.integers(1,d+1)); r2=int(rng.integers(1,d+1))
    KP=random_kraus(dY,dX,r1,rng); KQ=random_kraus(dY,dX,r2,rng)
    JP=sum(np.outer(vec(K),vec(K).conj()) for K in KP); JQ=sum(np.outer(vec(K),vec(K).conj()) for K in KQ)
    JO=JP+JQ
    # use a NON-orthonormal Kraus decomposition of Omega: V_k = sum_j A_kj K_j (linearly independent) via random invertible mixing of eigen-Kraus
    KO=kraus_from_choi(JO,dY,dX); r=len(KO)
    Mx=rng.normal(size=(r,r))+1j*rng.normal(size=(r,r))
    V=[sum(Mx[k,j]*KO[j] for j in range(r)) for k in range(r)]   # Omega(X)=sum_k V_k X V_k^* needs unitary mixing, so use unitary
    Q,_=np.linalg.qr(Mx); V=[sum(Q[k,j]*KO[j] for j in range(r)) for k in range(r)]
    W=np.array([vec(v) for v in V]).T
    Wp=np.linalg.pinv(W); T=herm(Wp@JP@Wp.conj().T)
    wt,U=np.linalg.eigh(T); wt=np.clip(wt,0,1); wt=np.where(wt<1e-9,0.0,wt); wt=np.where(wt>1-1e-9,1.0,wt)
    for name in ['geo','harm','log','mix','arith','skew']:
        cn=conns[name]; g=(U*cn.phi(wt))@U.conj().T
        # paper's formula: max_rho tr(g^T Omega^c(rho)),  Omega^c(rho)_{kl}=tr(V_k rho V_l^*)
        Mop=sum(g[k,l]*V[l].conj().T@V[k] for k in range(r) for l in range(r))
        lhs=np.linalg.eigvalsh(herm(Mop))[-1]
        Mc,_=mean_choi(JP,JQ,cn); KM=kraus_from_choi(Mc,dY,dX)
        rhs=np.linalg.norm(sum(K.conj().T@K for K in KM),2) if KM else 0.0
        worst=max(worst,abs(lhs-rhs))
print("max |max_rho tr(g(T)^T Omega^c(rho)) - ||Phi sigma Psi||_{1->1}| over 150 pairs, 6 connections, non-orthonormal Kraus frames:",worst)
