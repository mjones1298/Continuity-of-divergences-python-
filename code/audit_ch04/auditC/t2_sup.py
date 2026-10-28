import numpy as np
from lib import *
rng = np.random.default_rng(11)
f = lambda t: np.where(t>0, t*np.log(np.where(t>0,t,1)), 0.0)
h = h_of(f)
viol = 0; gaps=[]
for trial in range(60):
    dX = rng.integers(2,4); dY = rng.integers(1,4)
    r = rng.integers(1, dX*dY+1)
    Ks = rand_kraus(dX,dY,r); W = W_of(Ks)
    if np.linalg.matrix_rank(W)<r: continue
    spec = rng.uniform(0,0.9,size=r); T = rand_T(r, spec=spec)
    M = M_of(W,T,lambda t:h(np.asarray(t,float)),dY,dX)
    lam = np.linalg.eigvalsh(M)[-1]
    # rank-deficient pure input
    for k in range(1,dX):
        A = rng.normal(size=(dX,dX))+1j*rng.normal(size=(dX,dX))
        U,s,Vh = np.linalg.svd(A); s[k:] = 0; A = (U*s)@Vh; A/=np.linalg.norm(A)
        P1,P2 = phi_state(W,T,A,dY,dX)
        if np.trace(P2).real < 1e-12: continue
        # reduce to support
        val,res = Dhat_support(P1,P2,f,tol=1e-9)
        xi = (A@A.conj().T).T
        pred = np.trace(xi@M).real
        gaps.append(pred-val)
        if val > pred + 1e-8: viol+=1; print('VIOL lsc', val, pred)
        if val > lam+1e-8: viol+=1; print('VIOL sup', val, lam)
    # mixed state
    for _ in range(3):
        Z = rng.normal(size=(dX*dX, 3))+1j*rng.normal(size=(dX*dX,3))
        rho = Z@Z.conj().T; rho/=np.trace(rho).real
        C = None
        J1 = W@T@W.conj().T; J2 = W@(np.eye(r)-T)@W.conj().T
        # (Phi (x) id)(rho) = sum_i rho-component: use Choi: Phi_rho = Tr_{X}[ (J^{T_X}...)] simpler: rho = sum p_i u_i u_i^*
        w,V = np.linalg.eigh(rho)
        # direct via pure-state decomposition (linear):
        P1 = 0; P2 = 0
        for wi,vi in zip(w,V.T):
            Amat = vi.reshape(dX,dX)  # vec(A)=u => A = reshape
            C = np.kron(np.eye(dY), Amat.T)
            P1 = P1 + wi*C@J1@C.conj().T; P2 = P2 + wi*C@J2@C.conj().T
        val,res = Dhat_support(herm(P1),herm(P2),f,tol=1e-9)
        if val>lam+1e-8: viol+=1; print('VIOL mixed', val, lam)
print('violations', viol, 'min/max gap', min(gaps), max(gaps))
