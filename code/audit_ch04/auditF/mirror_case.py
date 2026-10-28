"""M1: Thm 12.4(d)(iv) mirror case and removal of 'not mutually singular'.
State level (dim X = 1, Y = C^4): rho1 = rho2 = (e1e1*+e2e2*)/2, c_Omega = 2.
Pair (rho1, rho2 + t e3e3*): rho2-part acquires a singular component, so D_f with f(0+)=+inf
(f=-log, f'(inf)=0 finite) must diverge as eta->0 although D_f(rho1||rho2) is finite.
Pair (rho1 + t e3e3*, rho2): D_f with f'(inf)=+inf (f = x log x) diverges likewise.
Also a mutually singular pair with Phi=0: (0, rho2) -> (t e3e3*, rho2): f = x log x, D_f(0||rho2)=0 finite, but D_f(t e3e3*||rho2)=+inf."""
import numpy as np
def psd_sqrt(M):
    w,V=np.linalg.eigh(M); return (V*np.sqrt(np.clip(w,0,None)))@V.conj().T
def Dhat(A,B,f,eta):
    n=A.shape[0]; A=A+eta*np.eye(n); B=B+eta*np.eye(n)
    Bh=psd_sqrt(B); Bi=np.linalg.inv(Bh); X=Bi@A@Bi; w,V=np.linalg.eigh((X+X.conj().T)/2)
    return float(np.trace(Bh@((V*f(w))@V.conj().T)@Bh).real) - n*eta*f(1.0)
E=lambda i: np.eye(4)[:,[i]]@np.eye(4)[[i],:]
r=(E(0)+E(1))/2
mlog=lambda x: -np.log(x); xlogx=lambda x: x*np.log(x)
for t in [1e-2,1e-4]:
    print("t=%g"%t)
    for eta in [1e-4,1e-8,1e-12]:
        print("  eta=%g  f=-log: D(rho1||rho2)=%.6f  D(rho1||rho2+tE3)=%.4f   | f=xlogx: D(rho1||rho2)=%.6f  D(rho1+tE3||rho2)=%.4f   | D(0||rho2)=%.2e  D(tE3||rho2)=%.4f"%(
          eta, Dhat(r,r,mlog,eta), Dhat(r,r+t*E(2),mlog,eta), Dhat(r,r,xlogx,eta), Dhat(r+t*E(2),r,xlogx,eta), Dhat(0*r,r,xlogx,eta), Dhat(t*E(2),r,xlogx,eta)))
