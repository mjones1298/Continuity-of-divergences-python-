import numpy as np
from lib import *
rng=np.random.default_rng(55)
worst={'1':0,'2':0,'4':0,'3':0}
def rpair(n,lo,hi,d):
    Q,_=np.linalg.qr(rng.normal(size=(n,n))+1j*rng.normal(size=(n,n)))
    A=(Q*rng.uniform(lo,hi,size=n))@Q.conj().T; A=herm(A)
    H=herm(rng.normal(size=(n,n))+1j*rng.normal(size=(n,n))); H*=d/np.linalg.norm(H,2)
    return A,herm(A+H)
for trial in range(20000):
    n=rng.integers(1,5); eps=10**rng.uniform(-3,0); d=10**rng.uniform(-5,-0.1)*eps
    A,B=rpair(n,eps,1+rng.uniform(0,2),d)
    wB=np.linalg.eigvalsh(B)
    if wB.min()<eps: continue
    nd=np.linalg.norm(A-B,2)
    L=np.linalg.norm(fun(A,np.log)-fun(B,np.log),2)/(nd/eps); worst['1']=max(worst['1'],L)
    beta=rng.uniform(0.01,1)
    L2=np.linalg.norm(fun(A,lambda t:t**-beta)-fun(B,lambda t:t**-beta),2)/(beta*eps**(-beta-1)*nd); worst['2']=max(worst['2'],L2)
    # (3),(4) need A,B in [0,1]
    A,B=rpair(n,0,1,rng.uniform(1e-4,0.5)*rng.choice([1,0.1]))
    wA=np.linalg.eigvalsh(A); wB=np.linalg.eigvalsh(B)
    if wB.min()<0 or wB.max()>1 or wA.min()<0 or wA.max()>1: continue
    nd=np.linalg.norm(A-B,2)
    xl=lambda t: np.where(t>0,t*np.log(np.where(t>0,t,1)),0)
    d3=np.linalg.norm(fun(A,xl)-fun(B,xl),2)
    bound=nd*np.log(1/nd)+(1+nd)*np.log(1+nd)
    worst['3']=max(worst['3'],d3/bound)
    d4=np.linalg.norm(fun(A,lambda t:np.clip(t,0,None)**(1+beta))-fun(B,lambda t:np.clip(t,0,None)**(1+beta)),2)
    worst['4']=max(worst['4'],d4/((3-beta)*nd))
print(worst)
