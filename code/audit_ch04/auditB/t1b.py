import sys; sys.path.insert(0,'.')
from lib import *
g = lambda t: np.sqrt(np.clip(t*(1-t),0,None))
sq = lambda x: np.sqrt(np.clip(x,0,None))
for eta in [1e-4,1e-6,1e-8,1e-10]:
    worst=0
    for trial in range(200):
        dX=rng.integers(1,4); dY=rng.integers(1,4)
        r=rng.integers(1,dX*dY+1)
        kr=rand_kraus(r,dY,dX); W=choi_w(kr,dX)
        mode=rng.integers(0,3)
        spec = rng.uniform(0,1,r) if mode==0 else (rng.choice([0,1],r).astype(float) if mode==1 else rng.choice([0,1,0.3],r))
        T1=effect(r,spec)
        J1=W@T1@W.conj().T; J2=W@(np.eye(r)-T1)@W.conj().T
        M=mean_reg(J1,J2,sq,eta)
        M2=W@psd_fun2(T1,g)@W.conj().T
        worst=max(worst,np.abs(M-M2).max())
    print(eta,worst)
