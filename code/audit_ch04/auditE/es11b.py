import numpy as np, sys
from lib import *
rng=np.random.default_rng(int(sys.argv[1]))
mode=sys.argv[2]   # 'both','thompson','hellinger'
lam_choice=sys.argv[3]  # e.g. 'sing' (exact 0/1/inner), 'near' (near 0/1 inner), 'gen'
dY,dX,r=int(sys.argv[5]),int(sys.argv[6]),int(sys.argv[7])
N=dY*dX

def unit(M):
    Q,_=np.linalg.qr(M);return Q

def build(x):
    p=0
    def take(n):
        nonlocal p;v=x[p:p+n];p+=n;return v
    W=(take(N*r)+1j*take(N*r)).reshape(N,r)
    U=unit((take(r*r)+1j*take(r*r)).reshape(r,r))
    lam_raw=take(r)
    if lam_choice=='sing':
        # first eigenvalue exactly 1, second exactly 0, third inner
        lam=np.array([1.0,0.0,1/(1+np.exp(-lam_raw[2]))])
    elif lam_choice=='near':
        e=np.exp(-abs(lam_raw))*0.1
        lam=np.array([1/(1+np.exp(-lam_raw[0]*0.3)),e[1],1-e[2]])
    else:
        lam=1/(1+np.exp(-lam_raw))
    C1=U@np.diag(lam)@U.conj().T; C2=np.eye(r)-C1
    Ts=[]
    for i in range(2):
        V=unit((take(r*r)+1j*take(r*r)).reshape(r,r)); t=1/(1+np.exp(-take(r)*2))
        Ts.append(V@np.diag(t)@V.conj().T)
    a1=U@np.diag(np.sqrt(lam))@U.conj().T; a2=U@np.diag(np.sqrt(1-lam))@U.conj().T
    P1=U@np.diag((lam>1e-12).astype(float))@U.conj().T; P2=U@np.diag(((1-lam)>1e-12).astype(float))@U.conj().T
    T1=P1@Ts[0]@P1; T2=P2@Ts[1]@P2
    A1=a1@T1@a1;B1=C1-A1;A2=a2@T2@a2;B2=C2-A2
    return W,[A1,A2],[B1,B2],[C1,C2]
nparam=2*N*r+2*r*r+r+2*(2*r*r+r)

def evaluate(x):
    W,A,B,C=build(x)
    # sum frame: Omega' = W W^*
    Wd=W
    J=lambda M: Wd@M@Wd.conj().T
    Js=[J(A[0]),J(B[0]),J(A[1]),J(B[1])]
    # fake Setup
    class S: pass
    s=S();s.W=Wd;s.r=r;s.dY=dY;s.dX=dX;s.A=A;s.B=B;s.C=C
    s.Om=[J(C[0]),J(C[1])]
    s.Omb=(s.Om[0]+s.Om[1])/2
    s.nOmb=opn(ptrace_Y(s.Omb,dY,dX));s.nOm=[opn(ptrace_Y(O,dY,dX)) for O in s.Om]
    s.Delta=lambda Z: ptrace_Y(Wd@Z@Wd.conj().T,dY,dX)
    # sec11 using exact projectors
    q=sec11(s,tol=1e-7)
    F1=Fhat(Js[0],Js[1],dY,dX);F2=Fhat(Js[2],Js[3],dY,dX)
    nO=s.nOmb
    th=np.sqrt(2)*nO*q['tanh_ac']; he=np.sqrt(nO)*q['hac']
    mm={'both':min(th,he),'thompson':th,'hellinger':he}[mode]
    R=nO*np.sqrt(q['tau_ac'])+0.5*q['ms']+s.nOm[0]*np.sqrt(q['kap1'])+s.nOm[1]*np.sqrt(q['kap2'])+mm
    return abs(F1-F2),R
def score(x):
    l,R=evaluate(x)
    return l/R if R>1e-12 else 0.0
best=0
for rest in range(int(sys.argv[4])):
    x=rng.standard_normal(nparam);fx=score(x);sig=0.5
    for it in range(6000):
        y=x+sig*rng.standard_normal(nparam)*(rng.random(nparam)<0.3)
        try: fy=score(y)
        except Exception: continue
        if fy>=fx: x,fx=y,fy;sig=min(sig*1.2,2)
        else: sig=max(sig*0.98,1e-4)
    best=max(best,fx);print(rest,fx,best,flush=True)
print('BEST',best)
