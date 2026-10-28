"""Theorem P1 constructive check: channels Psi_t with Kraus span span{K_j + t L_j}, TP restored by
C(t) = 1 - T_t^+(T_t(1) - 1) (T_t: C -> sum C_jk K_j(t)^* K_k(t)).  L_j chosen so that the map
c in C_A -> [sum c_j L_j] mod S is injective (A = Kraus-span intersection, S = K_Phi + K_Psi).
Checks: TP error, ||J(Psi_t)-J(Psi)||_1/dX, principal angles between ranges, Fhat."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))  # shared modules in code/common (glib.py)
import numpy as np, glib as g
from scipy.linalg import null_space, orth, subspace_angles
rng=np.random.default_rng(21)
def herm_basis(n):
    B=[]
    for i in range(n):
        for j in range(i,n):
            E=np.zeros((n,n),complex)
            if i==j: E[i,i]=1; B.append(E)
            else:
                E[i,j]=E[j,i]=1/np.sqrt(2); B.append(E.copy())
                F=np.zeros((n,n),complex); F[i,j]=-1j/np.sqrt(2); F[j,i]=1j/np.sqrt(2); B.append(F)
    return B
def tp_restore(Ks):
    s=len(Ks); dX=Ks[0].shape[1]; HB=herm_basis(s)
    cols=[sum(C[j,k]*Ks[j].conj().T@Ks[k] for j in range(s) for k in range(s)) for C in HB]
    hx=herm_basis(dX)
    A=np.array([[np.trace(h.conj().T@c).real for c in cols] for h in hx])
    rhs=np.array([np.trace(h.conj().T@(np.eye(dX)-sum(K.conj().T@K for K in Ks))).real for h in hx])
    coef=np.linalg.pinv(A)@rhs
    C=np.eye(s)+sum(a*h for a,h in zip(coef,HB))
    return C, np.linalg.matrix_rank(A,tol=1e-10)==len(hx)
def channel_from(Ks,C):
    w,U=np.linalg.eigh(g.herm(C)); assert w.min()>0
    R=U*np.sqrt(w)
    return [sum(np.conj(R[j,m])*Ks[j] for j in range(len(Ks))) for m in range(len(Ks))]
def kraus_indep(Ks):
    J=g.choi(Ks); W=g.frame(J); dY,dX=Ks[0].shape
    return [W[:,k].reshape(dY,dX) for k in range(W.shape[1])]
def run(name, KPhi, KPsi, ts=(1e-2,1e-3,1e-4)):
    dY,dX=KPsi[0].shape
    KPsi=kraus_indep(KPsi)
    JPhi,JPsi=g.choi(KPhi),g.choi(KPsi)
    S=orth(np.hstack([orth(JPhi,rcond=1e-10),orth(JPsi,rcond=1e-10)]),rcond=1e-10)
    Sperp=null_space(S.conj().T)
    RPhi=orth(JPhi,rcond=1e-10)
    s=len(KPsi); Mpsi=np.array([g.vec(K) for K in KPsi]).T
    # C_A = {c : Mpsi c in range J(Phi)}
    Mres=(np.eye(dX*dY)-RPhi@RPhi.conj().T)@Mpsi; u_,sv_,vh_=np.linalg.svd(Mres); CA=vh_[np.r_[sv_,np.zeros(s)][:s]<1e-9].conj().T
    a=CA.shape[1]; cO=Sperp.shape[1]
    _,reg=tp_restore(KPsi)
    print(f"== {name}: rank J(Phi)={RPhi.shape[1]}, rank J(Psi)={s}, rank J(Om)={S.shape[1]}, a={a}, c_Om={cO}, Psi^c injective={reg}, Fhat(Phi,Psi)={g.Fhat(JPhi,JPsi,dY,dX):.4f}")
    part = a>cO
    if part: print("   a > c_Om: partial separation (rank c_Om tilt)")
    # Lambda: C^s -> S^perp, injective on C_A
    Lam=np.zeros((dX*dY,s),complex)
    if a>0:
        Q=np.linalg.qr(np.hstack([CA,rng.normal(size=(s,s))]))[0][:, :s]  # orthonormal basis starting with C_A
        Gm=rng.normal(size=(cO,a))+1j*rng.normal(size=(cO,a))
        Lam=Sperp@Gm@CA.conj().T
    L=[Lam[:,j].reshape(dY,dX) for j in range(s)]
    for t in ts:
        Kt=[K+t*Lj for K,Lj in zip(KPsi,L)]
        C,_=tp_restore(Kt); Kc=channel_from(Kt,C)
        tp=np.abs(sum(K.conj().T@K for K in Kc)-np.eye(dX)).max()
        Jt=g.choi(Kc); dist=np.abs(np.linalg.eigvalsh(Jt-JPsi)).sum()/1
        ang=subspace_angles(orth(Jt,rcond=1e-10),RPhi).min() if a>0 else np.nan
        print(f"   t={t:g}: TPerr={tp:.1e} ||C-1||={np.linalg.norm(C-np.eye(s),2):.2e} ||dJ||_1={dist:.2e} min angle(ranJ(Phi),ranJ(Psi_t))={ang:.2e} Fhat(Phi,Psi_t)={g.Fhat(JPhi,Jt,dY,dX):.2e}")
def ad(gm): return [np.diag([1,np.sqrt(1-gm)]).astype(complex), np.sqrt(gm)*np.array([[0,1],[0,0]],complex)]
run("AD(0.3) vs AD(0.3)", ad(0.3), ad(0.3))
X=np.array([[0,1],[1,0]],complex); Z=np.diag([1.,-1]).astype(complex)
deph=lambda q,Ks: [np.sqrt(1-q)*K for K in Ks]+[np.sqrt(q)*Z@K for K in Ks]
run("AD(0.3)+deph(0.1) [T1+T2] vs itself", deph(0.1,ad(0.3)), deph(0.1,ad(0.3)))
run("AD(0.2) vs AD(0.5)", ad(0.2), ad(0.5))
# measure-and-prepare channel on C^4 (rotations fail for (Phi,Phi))
d=4; e=np.eye(d)
MP=[np.outer(e[i],e[0])/np.sqrt(d) for i in range(d)]+[np.outer(e[0],e[j]) for j in range(1,d)]
run("measure-prepare d=4 vs itself", MP, MP)
# rotation test for measure-prepare: is {K in S: HK - KG in S} nontrivial for random H,G?
S=orth(np.array([g.vec(K) for K in MP]).T); Sp=null_space(S.conj().T)
worst=10
for _ in range(20):
    H=g.herm(rng.normal(size=(d,d))+1j*rng.normal(size=(d,d))); G=g.herm(rng.normal(size=(d,d))+1j*rng.normal(size=(d,d)))
    Mmap=np.array([Sp.conj().T@g.vec(H@S[:,k].reshape(d,d)-S[:,k].reshape(d,d)@G) for k in range(S.shape[1])]).T
    worst=min(worst,np.linalg.svd(Mmap,compute_uv=False).min())
print("   measure-prepare: min singular value of K->[HK-KG] on S over random (H,G):",worst,"(0 => rotations cannot separate)")
# random pairs
for (dY,dX,k1,k2) in [(2,2,2,2),(3,2,2,3),(2,3,3,3),(3,3,4,4)]:
    run(f"random dY={dY} dX={dX} ranks {k1},{k2}", g.rand_kraus(rng,dY,dX,k1), g.rand_kraus(rng,dY,dX,k2), ts=(1e-3,))
