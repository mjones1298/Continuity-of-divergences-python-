import sys; sys.path.insert(0,'.')
from lib import *
import mplib as M
import mpmath as mp
al=mp.mpf('0.3')
def mxlogx(x): return x*mp.log(x) if x>0 else mp.mpf(0)
fs = {
 'Hellinger': (lambda x:(1-mp.sqrt(x))**2, lambda t: 1-2*np.sqrt(max(t*(1-t),0))),
 'Hellinger_a': (lambda x:(1-x**al)/(1-al), lambda t: ((1-t)-t**0.3*(1-t)**0.7)/0.7),
 'LC': (lambda x:(x-1)**2/(x+1), lambda t:(2*t-1)**2),
 'JS': (lambda x: mxlogx(x)-(1+x)*mp.log((1+x)/2), lambda t: np.log(2)+(t*np.log(t) if t>0 else 0)+((1-t)*np.log(1-t) if t<1 else 0)),
 'sqrt_neg': (lambda x: -mp.sqrt(x), lambda t: -np.sqrt(max(t*(1-t),0))),
}
def hf_apply(T,hf):
    w,V=eigh(herm(T)); w=np.clip(w,0,1)
    return (V*np.array([hf(x) for x in w]))@V.conj().T
worst={k:0 for k in fs}
for trial in range(6):
    dX=rng.integers(1,3); dY=rng.integers(1,3)
    r=rng.integers(1,dX*dY+1)
    kr=rand_kraus(r,dY,dX); W=choi_w(kr,dX)
    mode=trial%3
    spec = rng.uniform(0.05,0.95,r) if mode==0 else (rng.choice([0,1],r).astype(float) if mode==1 else rng.choice([0,1,0.3,0.7],r))
    T=effect(r,spec)
    JPhi=W@T@W.conj().T; JPsi=W@(np.eye(r)-T)@W.conj().T
    for name,(f,hf) in fs.items():
        Mx=herm(ptrace_out(W@hf_apply(T,hf)@W.conj().T,dY,dX))
        for k in range(2):
            A=rand_c(dX,dX); A/=np.linalg.norm(A)
            C=np.kron(np.eye(dY),A.T)
            Pa=C@JPhi@C.conj().T; Pb=C@JPsi@C.conj().T
            d=float(M.persp_trace(M.tomp(Pa),M.tomp(Pb),f,mp.mpf('1e-12')))
            pred=np.trace((A@A.conj().T).T@Mx).real
            worst[name]=max(worst[name],abs(d-pred))
    print(trial,dX,dY,r,spec.round(2),worst,flush=True)
