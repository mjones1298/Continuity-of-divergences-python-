import numpy as np
from lib import *
rng=np.random.default_rng(101)
def wgmean_frame(A,B,beta):
    # A #_beta B = W (T #_beta (1-T)) W^*,  T-derivative of A wrt A+B: g(t)=t^{1-beta}(1-t)^beta
    W=frame(A+B);Wp=np.linalg.pinv(W)
    T=herm(Wp@A@Wp.conj().T)
    g=fun_herm(T,lambda t:np.clip(t,0,1)**(1-beta)*np.clip(1-t,0,1)**beta)
    return herm(W@g@W.conj().T)
def Qhat(JP,JQ,al,dY,dX):
    # lambda_min Tr_Y( J(Psi) #_al J(Phi) )
    return lmin(ptrace_Y(wgmean_frame(JQ,JP,al),dY,dX))
def same_supp(dY,dX,m,rng,spread):
    N=dY*dX
    U,_=np.linalg.qr(rng.standard_normal((N,N))+1j*rng.standard_normal((N,N)));U=U[:,:m]
    return [U@rand_psd(m,m,rng,np.exp(spread*rng.standard_normal()))@U.conj().T for _ in range(4)]
wb=0;wc=0;cnt=0
for trial in range(4000):
    dY,dX=[(2,2),(3,2),(2,3),(1,3),(3,1)][rng.integers(5)];N=dY*dX
    m=rng.integers(1,N+1)
    sp=rng.choice([0.1,0.7,2.0])
    J=same_supp(dY,dX,m,rng,sp)
    # make second pair close to first sometimes
    if rng.random()<0.5:
        e=np.exp(-rng.uniform(0.5,8))
        U=np.linalg.qr(J[0]+J[1]+J[2]+J[3]+1e-12*np.eye(N))[0]
        P=[j for j in J]
        P[2]=J[0]+e*J[2];P[3]=J[1]+e*J[3]
        J=P
    S=Setup(*J,dY,dX)
    if S.gamma>0.5-1e-6: continue
    cnt+=1
    c1=S.C[0];w,V=eigh(herm(c1))
    a1=fun_herm(c1,np.sqrt);a2=fun_herm(S.C[1],np.sqrt)
    T1=herm(np.linalg.inv(a1)@S.A[0]@np.linalg.inv(a1));T2=herm(np.linalg.inv(a2)@S.A[1]@np.linalg.inv(a2))
    tau=opn(T1-T2);th=2*S.gamma
    F1=Fhat(J[0],J[1],dY,dX);F2=Fhat(J[2],J[3],dY,dX)
    rb=abs(F1-F2)/(S.nOmb*(np.sqrt(tau)+np.sqrt(2)*th)) if tau+th>1e-12 else 0
    wb=max(wb,rb)
    al=rng.uniform(0.05,0.95)
    Q1=Qhat(J[0],J[1],al,dY,dX);Q2=Qhat(J[2],J[3],al,dY,dX)
    rc=abs(Q1-Q2)/(S.nOmb*(tau**al+tau**(1-al)+2*np.sqrt(2)*th)) if tau+th>1e-12 else 0
    wc=max(wc,rc)
print(cnt,'worst (b)',wb,'worst (c)',wc)
