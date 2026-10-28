import sys; sys.path.insert(0,'.')
from lib import *
from scipy.linalg import eigh
al=0.3
def xlogx(x): 
    x=np.asarray(x,float); out=np.zeros_like(x); m=x>0; out[m]=x[m]*np.log(x[m]); return out
fs = {
 'Hellinger': (lambda x:(1-np.sqrt(np.clip(x,0,None)))**2, lambda t: 1-2*np.sqrt(np.clip(t*(1-t),0,None))),
 'Hellinger_a': (lambda x:(1-np.clip(x,0,None)**al)/(1-al), lambda t: ((1-t)-np.clip(t,0,1)**al*np.clip(1-t,0,1)**(1-al))/(1-al)),
 'LC': (lambda x:(x-1)**2/(x+1), lambda t:(2*t-1)**2),
 'JS': (lambda x: xlogx(x)-(1+x)*np.log((1+x)/2), lambda t: np.log(2)+xlogx(t)+xlogx(1-t)),
 # f(1) != 0 examples
 'sqrt_neg': (lambda x: -np.sqrt(np.clip(x,0,None)), lambda t: -np.sqrt(np.clip(t*(1-t),0,None))),
 'const+sqrt': (lambda x: 2-np.sqrt(np.clip(x,0,None)), lambda t: 2*(1-t)-np.sqrt(np.clip(t*(1-t),0,None))),
}
worst={k:0 for k in fs}
for trial in range(12):
    dX=rng.integers(1,4); dY=rng.integers(1,4)
    r=rng.integers(1,dX*dY+1)
    kr=rand_kraus(r,dY,dX); W=choi_w(kr,dX)
    mode=trial%3
    spec = rng.uniform(0.05,0.95,r) if mode==0 else (rng.choice([0,1],r).astype(float) if mode==1 else rng.choice([0,1,0.3,0.7],r))
    T=effect(r,spec)
    JPhi=W@T@W.conj().T; JPsi=W@(np.eye(r)-T)@W.conj().T
    for name,(f,hf) in fs.items():
        M=ptrace_out(W@psd_fun2(T,hf)@W.conj().T,dY,dX)
        M=herm(M)
        # random invertible A
        for k in range(2):
            A=rand_c(dX,dX); A/=np.linalg.norm(A)
            C=np.kron(np.eye(dY),A.T)
            Pa=C@JPhi@C.conj().T; Pb=C@JPsi@C.conj().T
            d=Df_reg(Pa,Pb,f,eta=1e-10)
            pred=np.trace((A@A.conj().T).T@M).real
            worst[name]=max(worst[name],abs(d-pred))
print(worst)
