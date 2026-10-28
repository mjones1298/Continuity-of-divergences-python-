import numpy as np, sys
from lib import *
rng=np.random.default_rng(int(sys.argv[1]))
dY,dX,r=int(sys.argv[2]),int(sys.argv[3]),int(sys.argv[4])
N=dY*dX
def unit(M):
    Q,_=np.linalg.qr(M);return Q
def build(x):
    p=0
    def take(n):
        nonlocal p;v=x[p:p+n];p+=n;return v
    W=(take(N*r)+1j*take(N*r)).reshape(N,r)
    Gs=[]
    for i in range(4):
        L=(take(r*r)+1j*take(r*r)).reshape(r,r);Gs.append(L@L.conj().T)
    S=sum(Gs);Sih=fun_herm(S,lambda w:1/np.sqrt(np.clip(w,1e-300,None)))
    P=[herm(Sih@G@Sih) for G in Gs]
    return W,P
nparam=2*N*r+4*2*r*r
def score(x):
    W,P=build(x)
    J=[W@M@W.conj().T for M in P]
    Om=[J[0]+J[1],J[2]+J[3]]
    nOmb=opn(ptrace_Y((Om[0]+Om[1])/2,dY,dX))
    A1,B1,A2,B2=P
    dl=max(opn(A1-A2),opn(B1-B2))
    if dl<1e-9 or nOmb<1e-9: return 0.0
    F1=Fhat(J[0],J[1],dY,dX);F2=Fhat(J[2],J[3],dY,dX)
    return abs(F1-F2)/(nOmb*np.sqrt(dl))
best=0
for rest in range(int(sys.argv[5])):
    x=rng.standard_normal(nparam);fx=score(x);sig=0.5
    for it in range(4000):
        y=x+sig*rng.standard_normal(nparam)*(rng.random(nparam)<0.3)
        fy=score(y)
        if fy>=fx: x,fx=y,fy;sig=min(sig*1.2,2)
        else: sig=max(sig*0.98,1e-4)
    best=max(best,fx);print(rest,fx,best,flush=True)
print('BEST',best,'claimed 11/4=2.75')
