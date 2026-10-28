import sys; sys.path.insert(0,'.')
from lib import *
from scipy.optimize import minimize
import mplib as M, mpmath as mp
def Qalpha_state(A,B,alpha):
    # tr B #_alpha A = tr B^{1/2}(B^{-1/2} A B^{-1/2})^alpha B^{1/2}  (B>0,A>=0)
    return np.trace(perspective_pd(A,B,lambda x: np.clip(x,0,None)**alpha)).real
alpha=0.35
for trial in range(5):
    dX=2; dY=2
    r=dX*dY
    kr=rand_kraus(r,dY,dX); W=choi_w(kr,dX)
    T=effect(r)
    JPhi=W@T@W.conj().T; JPsi=W@(np.eye(r)-T)@W.conj().T
    h=lambda t: np.clip(t,0,1)**alpha*np.clip(1-t,0,1)**(1-alpha)
    Mx=herm(ptrace_out(W@psd_fun2(T,h)@W.conj().T,dY,dX))
    lam=np.linalg.eigvalsh(Mx)[0]
    # direct: minimise over pure input states u=vec(A)
    def obj(x):
        A=(x[:dX*dX]+1j*x[dX*dX:]).reshape(dX,dX); A=A/np.linalg.norm(A)
        C=np.kron(np.eye(dY),A.T)
        Pa=C@JPhi@C.conj().T; Pb=C@JPsi@C.conj().T
        # use (Psi first): Q = tr Psi_u #_alpha Phi_u = tr B^{1/2} (B^{-1/2} A B^{-1/2})^alpha B^{1/2}, B=Psi_u, A=Phi_u
        eta=1e-12
        return Qalpha_state(Pa+eta*np.eye(len(Pa)),Pb+eta*np.eye(len(Pb)),alpha)
    best=1e9
    for s in range(6):
        res=minimize(obj,rng.normal(size=2*dX*dX),method='Nelder-Mead',options={'xatol':1e-9,'fatol':1e-12,'maxiter':4000})
        best=min(best,res.fun)
    print(trial,"lambda_min formula",lam,"direct inf over pure states",best)
