import numpy as np, mpmath as mp, sys
from qlib import *
from scipy.optimize import minimize
mp.mp.dps=60
rng=np.random.default_rng(int(sys.argv[1]) if len(sys.argv)>1 else 0)
def Fh(JA,JB):
    # rank-aware: if both full rank use eps=0, else eps=1e-30 in 70 digits
    full = min(np.linalg.eigvalsh(JA)[0]/np.linalg.eigvalsh(JA)[-1],np.linalg.eigvalsh(JB)[0]/np.linalg.eigvalsh(JB)[-1])>1e-13
    return float(fid_mp(JA,JB,2,2,'0',dps=60)) if full else fid_sub(JA,JB,2,2)
def psd_sqrt_inv(C):
    w,V=np.linalg.eigh(C); return V@np.diag(w**-0.5)@V.conj().T
def bound_terms(J1,K1,J2,K2):
    Jo=J1+K1+J2+K2
    w,V=np.linalg.eigh(Jo); keep=w>1e-10*w.max()
    W=V[:,keep]@np.diag(np.sqrt(w[keep]))   # frame: columns vec(V_k)
    Wp=np.linalg.pinv(W)
    ch=lambda J: Wp@J@Wp.conj().T
    A1,B1,A2,B2=ch(J1),ch(K1),ch(J2),ch(K2)
    assert np.allclose(A1+B1+A2+B2,np.eye(W.shape[1]),atol=1e-8)
    C1,C2=A1+B1,A2+B2
    T1=psd_sqrt_inv(C1)@A1@psd_sqrt_inv(C1); T2=psd_sqrt_inv(C2)@A2@psd_sqrt_inv(C2)
    tau=np.linalg.norm(T1-T2,2); th=np.linalg.norm(C1-C2,2)
    # ||Omega_bar^*(1)||: Omega_bar = Omega'/2 ; Omega'^*(1) = Tr_Y J^T -> for Choi, Omega^*(1)^T = Tr_Y J
    nrm=np.linalg.norm(ptrace_Y(Jo,2,2),2)/2
    return tau,th,nrm,np.linalg.eigvalsh(C1)[0],np.linalg.eigvalsh(C2)[0]
def check(J1,K1,J2,K2):
    tau,th,nrm,c1,c2=bound_terms(J1,K1,J2,K2)
    lhs=abs(Fh(J1,K1)-Fh(J2,K2)); rhs=nrm*(np.sqrt(tau)+np.sqrt(2)*th)
    return lhs,rhs,tau,th,nrm,min(c1,c2)
def randJ(r): return choi_from_kraus(rand_channel_kraus(2,2,r,rng))
# perturbation inside a fixed support S (columns G) preserving TP:
def tp_kernel_basis(G):
    r=G.shape[1]; basis=[]
    # hermitian basis of r x r
    Hs=[]
    for i in range(r):
        for j in range(i,r):
            if i==j: M=np.zeros((r,r),complex); M[i,i]=1; Hs.append(M)
            else:
                M=np.zeros((r,r),complex); M[i,j]=M[j,i]=1; Hs.append(M)
                M=np.zeros((r,r),complex); M[i,j]=1j; M[j,i]=-1j; Hs.append(M)
    L=np.array([np.concatenate([ptrace_Y(G@M@G.conj().T,2,2).real.ravel(),ptrace_Y(G@M@G.conj().T,2,2).imag.ravel()]) for M in Hs]).T
    _,s,Vt=np.linalg.svd(L); null=Vt[(s>1e-10).sum():]
    return [sum(c*M for c,M in zip(v,Hs)) for v in null]
def perturb_in_support(J,S_G,Ks,delta):
    # J = G X G^*, add delta * G H G^* with H in TP-kernel; keep PSD by shrinking
    Gp=np.linalg.pinv(S_G); Xc=Gp@J@Gp.conj().T
    H=sum(rng.normal()*K for K in Ks); H/=np.linalg.norm(H,2)
    for _ in range(60):
        Xn=Xc+delta*H
        if np.linalg.eigvalsh(Xn)[0]>=-1e-13: break
        delta/=2
    return S_G@Xn@S_G.conj().T
results=[]
def record(tag,J1,K1,J2,K2):
    lhs,rhs,tau,th,nrm,cm=check(J1,K1,J2,K2)
    results.append((lhs/rhs if rhs>0 else (np.inf if lhs>1e-12 else 0),lhs,rhs,tau,th,nrm,tag))
# family 1: random full-rank, small perturbations within full support
for k in range(80):
    r1,r2=rng.choice([1,2,3,4],2)
    J1,K1=randJ(r1),randJ(r2)
    Jo=J1+K1; w,V=np.linalg.eigh(Jo); keep=w>1e-10; G=V[:,keep]
    Ks=tp_kernel_basis(G)
    if not Ks: continue
    d=10**rng.uniform(-4,0)
    J2=perturb_in_support(J1,G,Ks,d); K2=perturb_in_support(K1,G,Ks,d)
    # ensure support of J2+K2 equals that of J1+K1
    if np.linalg.matrix_rank(J2+K2,1e-9)!=keep.sum(): continue
    record(f'rand r=({r1},{r2}) d={d:.1e}',J1,K1,J2,K2)
# family 2: mixtures Phi2=(1-s)Phi1+s Psi1 etc (F2 near 1)
for k in range(60):
    r1,r2=rng.choice([1,2],2); J1,K1=randJ(r1),randJ(r2)
    s,t=rng.uniform(0,1,2)*10**rng.uniform(-3,0)
    J2=(1-s)*J1+s*K1; K2=(1-t)*K1+t*J1
    record(f'mix r=({r1},{r2}) s={s:.1e}',J1,K1,J2,K2)
# family 3: close channels (F near 1) perturbed
for k in range(60):
    r=rng.choice([1,2,3,4]); J1=randJ(r)
    U=rand_channel_kraus(2,2,1,rng)[0]
    Ks=kraus_from_choi(J1,2,2)
    e=10**rng.uniform(-3,-0.5)
    from scipy.linalg import expm
    Hh=rng.normal(size=(2,2))+1j*rng.normal(size=(2,2)); Hh=(Hh+Hh.conj().T)/2
    Ue=expm(-1j*e*Hh)
    K1=choi_from_kraus([Ue@K for K in Ks])
    Jo=J1+K1; w,V=np.linalg.eigh(Jo); keep=w>1e-10*w.max(); G=V[:,keep]; Kb=tp_kernel_basis(G)
    if not Kb: continue
    d=10**rng.uniform(-4,-1)
    J2=perturb_in_support(J1,G,Kb,d); K2=perturb_in_support(K1,G,Kb,d)
    if np.linalg.matrix_rank(J2+K2,1e-9)!=np.linalg.matrix_rank(Jo,1e-9): continue
    record(f'near r={r} e={e:.1e} d={d:.1e}',J1,K1,J2,K2)
# family 4: replacement-channel family of C1 and depolarized variants
P=lambda v: np.outer(v,np.conj(v))
for th in [1e-3,1e-2,0.1,0.5,1.0,1.5]:
    J0=np.kron(P(np.array([1,0])),I2); Jt=np.kron(P(np.array([np.cos(th),np.sin(th)])),I2)
    for s in [1e-3,1e-2,0.1,0.5]:
        record(f'C1-type th={th} s={s}',J0,Jt,(1-s)*J0+s*Jt,(1-s)*Jt+s*J0)
# family 5: noisy gates of C4
def JN(p,th):
    v=vec(np.diag([np.exp(-1j*th),np.exp(1j*th)])@H); return (1-p)*np.outer(v,v.conj())+p/2*np.eye(4)
for p in [1e-1,1e-2,1e-3]:
    for th in [1e-3,1e-2,1e-1]:
        for dp in [1e-4,1e-3]:
            record(f'gate p={p} th={th} dp={dp}',JN(p,0),JN(p,th),JN(p+dp,0),JN(p,th*1.1))
results.sort(key=lambda x:-x[0])
print('N instances',len(results))
for r in results[:12]: print(f"ratio={r[0]:.4f} lhs={r[1]:.3e} rhs={r[2]:.3e} tau={r[3]:.2e} tanh={r[4]:.2e} ||Ob*(1)||={r[5]:.3f}  {r[6]}")
print('violations:',sum(r[0]>1 for r in results))
from pathlib import Path
np.save(Path(__file__).resolve().parent / f'c6_results_{sys.argv[1] if len(sys.argv)>1 else 0}.npy',np.array([r[:6] for r in results]))
