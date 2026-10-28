import sys; sys.path.insert(0,'.')
from lib import *
def xlogx(x): return np.where(x>1e-300,x*np.log(np.where(x>1e-300,x,1)),0.)
def bound(d): return d*np.log(1/d)+(1+d)*np.log(1+d)
w3=0; w4={0.2:0,0.5:0,0.9:0,1.0:0}
for t in range(20000):
    n=rng.integers(1,4)
    def rE():
        Q,_=np.linalg.qr(rand_c(n,n)); spec=np.where(rng.random(n)<0.4,rng.choice([0.,1.],n),rng.uniform(0,1,n))
        return herm(Q@np.diag(spec)@Q.conj().T)
    A=rE()
    s=10**rng.uniform(-3,0)
    H=herm(rand_c(n,n)); H*= s/opnorm(H)
    B=herm(A+H)
    w,V=eigh(B); B=(V*np.clip(w,0,1))@V.conj().T
    d=opnorm(A-B)
    if d<1e-9 or d>1: continue
    r3=opnorm(psd_fun2(A,xlogx)-psd_fun2(B,xlogx))/bound(d); w3=max(w3,r3)
    for b in w4:
        f=lambda x: np.clip(x,0,None)**(1+b)
        w4[b]=max(w4[b],opnorm(psd_fun2(A,f)-psd_fun2(B,f))/((3-b)*d))
print("Lemma 6.9(3) max ratio",w3,"(4) max ratios",w4)
