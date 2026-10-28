from pathlib import Path
# K4: (1/n) D((N_p)^{⊗n}(psi) || (N_{p,δ})^{⊗n}(psi)) / δ^2 via BKM metric (2nd order), optimized over pure inputs with full ancilla
import numpy as np, scipy.optimize as so, sys
from lib import *
def kraus_N(p,delta):
    V=Rz(delta)@H
    return [np.sqrt(1-3*p/4)*V]+[np.sqrt(p/4)*P@V for P in (X,Y,Z)]
def apply_n(Ks_list,psi,n):
    # psi: vector on (2^n system) ⊗ (2^n ancilla); system qubits first
    rho=np.outer(psi,psi.conj())
    dA=2**n
    for k in range(n):
        Ks=Ks_list[k]
        new=np.zeros_like(rho)
        for K in Ks:
            Kf=np.kron(np.kron(np.eye(2**k),K),np.eye(2**(n-k-1)*dA))
            new+=Kf@rho@Kf.conj().T
        rho=new
    return rho
def bkm(rho,drho):
    w,V=np.linalg.eigh(rho); w=np.clip(w,1e-300,None)
    A=V.conj().T@drho@V
    L=np.log(w)
    W=np.subtract.outer(w,w); LL=np.subtract.outer(L,L)
    with np.errstate(divide='ignore',invalid='ignore'):
        K=np.where(np.abs(W)>1e-14*np.maximum.outer(w,w),LL/W,1/np.add.outer(w,w)*2)  # limit 1/w
    K=np.where(np.abs(W)>1e-14*np.maximum.outer(w,w),K,1/w[:,None]*np.ones_like(K))
    return np.real(np.sum(np.abs(A)**2*K))
def D_over_delta2(psi,p,n,h=1e-6):
    K0=kraus_N(p,0.0)
    rho=apply_n([K0]*n,psi,n)
    # derivative wrt delta by central differences of Kraus (linear in channel -> exact up to O(h^2))
    Kp=kraus_N(p,h); Km=kraus_N(p,-h)
    drho=(apply_n([Kp]*n,psi,n)-apply_n([Km]*n,psi,n))/(2*h)
    return 0.5*bkm(rho,drho)/n
p=float(sys.argv[1]); n=int(sys.argv[2]); seeds=int(sys.argv[3])
rng=np.random.default_rng(7)
dim=4**n
def f(x):
    psi=x[:dim]+1j*x[dim:]; psi=psi/np.linalg.norm(psi)
    return -D_over_delta2(psi,p,n)
# max-entangled reference
me=np.zeros(dim,complex)
for i in range(2**n): me[i*2**n+i]=1
me/=np.linalg.norm(me)
print(f'p={p} n={n}: max-ent input (1/n)D/δ² = {-f(np.concatenate([me.real,me.imag])):.6g}   ln(1/p)={np.log(1/p):.4g}')
best=None
for s in range(seeds):
    x0=rng.normal(size=2*dim)
    r=so.minimize(f,x0,method='BFGS',options={'maxiter':3000,'gtol':1e-9})
    if best is None or r.fun<best.fun: best=r
print(f'p={p} n={n}: best (1/n)D/δ² = {-best.fun:.6g};  times p = {-best.fun*p:.5g}  (claim >= 0.1496)')
np.save(Path(__file__).resolve().parent / f'k4_best_{p}_{n}.npy',best.x)
