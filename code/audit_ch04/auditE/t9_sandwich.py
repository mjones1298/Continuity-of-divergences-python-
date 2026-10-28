import numpy as np
from lib import *
rng=np.random.default_rng(31)
def rand_four(r,rng,spread):
    # random A1,B1,A2,B2 PSD with sum identity; spread controls near-singularity of C_i
    Gs=[]
    for _ in range(4):
        G=rand_psd(r,rng.integers(1,r+1),rng,scale=np.exp(spread*rng.standard_normal()))
        Gs.append(G)
    S=sum(Gs);Sih=fun_herm(S,lambda w:1/np.sqrt(w))
    return [herm(Sih@G@Sih) for G in Gs]
fams={'t':(lambda t:t,lambda d:d,1.0),'2t-1':(lambda t:2*t-1,lambda d:2*d,1.0),'t2':(lambda t:t**2,lambda d:2*d,1.0),
      'gs':(lambda t:np.sqrt(np.clip(t*(1-t),0,None)),lambda d:np.sqrt(d),0.5),'one':(lambda t:np.ones_like(t),lambda d:0*d,1.0),
      'LeCam':(lambda t:(2*t-1)**2,lambda d:4*d,1.0)}
worst={k:0 for k in fams};cnt=0
for trial in range(20000):
    r=rng.integers(1,5)
    A1,B1,A2,B2=rand_four(r,rng,rng.choice([0.1,1,3]))
    C1=A1+B1;C2=A2+B2
    w=np.linalg.eigvalsh(C1)
    if w[0]<1e-7 or w[-1]>1-1e-7: continue
    gam=opn(C1-0.5*np.eye(r))
    a1=fun_herm(C1,np.sqrt);a2=fun_herm(C2,np.sqrt)
    c1i=fun_herm(C1,lambda x:1/np.sqrt(x));c2i=fun_herm(C2,lambda x:1/np.sqrt(x))
    T1=herm(c1i@A1@c1i);T2=herm(c2i@A2@c2i)
    tau=opn(T1-T2)
    cnt+=1
    for k,(f,om,nf) in fams.items():
        lhs=opn(a1@fun_herm(T1,f)@a1-a2@fun_herm(T2,f)@a2)
        rhs=0.5*om(tau)+2*np.sqrt(2)*gam*nf
        if rhs>1e-12: worst[k]=max(worst[k],lhs/rhs)
print(cnt,worst)
