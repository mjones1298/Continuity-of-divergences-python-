import sys; sys.path.insert(0,'.')
from lib import *
def xlogx(x): return np.where(x>0,x*np.log(np.where(x>0,x,1)),0.)
worstQ=0; worstH=0; worstLC=0
for trial in range(2000):
    dX=rng.integers(1,4); dY=rng.integers(1,4)
    r=rng.integers(1,dX*dY+1)
    kr=rand_kraus(r,dY,dX); W=choi_w(kr,dX)
    Om=opnorm(sum(K.conj().T@K for K in kr))
    def rT():
        mode=rng.integers(0,4)
        if mode==0: spec=rng.uniform(0,1,r)
        elif mode==1: spec=rng.choice([0,1],r).astype(float)
        elif mode==2: spec=np.clip(rng.normal(0.5,0.4,r),0,1)
        else: spec=rng.choice([0,1,0.5],r)
        return effect(r,spec)
    T1=rT()
    T2 = rT() if rng.random()<0.5 else herm(T1+ rng.uniform(0,0.2)*(effect(r)-0.5*np.eye(r)))
    # clip T2 into [0,1]
    w,V=eigh(herm(T2)); T2=(V*np.clip(w,0,1))@V.conj().T
    d=opnorm(T1-T2)
    if d<1e-9: continue
    a=rng.uniform(0.05,0.95)
    h=lambda t: np.clip(t,0,1)**a*np.clip(1-t,0,1)**(1-a)
    q=lambda T: np.linalg.eigvalsh(herm(ptrace_out(W@psd_fun2(T,h)@W.conj().T,dY,dX)))[0]
    worstQ=max(worstQ,abs(q(T1)-q(T2))/(Om*(d**a+d**(1-a))))
    hH=lambda t: 1-2*np.sqrt(np.clip(t*(1-t),0,None))
    DH=lambda T: np.linalg.eigvalsh(herm(ptrace_out(W@psd_fun2(T,hH)@W.conj().T,dY,dX)))[-1]
    worstH=max(worstH,abs(DH(T1)-DH(T2))/(Om*2*np.sqrt(d)))
    hL=lambda t:(2*t-1)**2
    DL=lambda T: np.linalg.eigvalsh(herm(ptrace_out(W@psd_fun2(T,hL)@W.conj().T,dY,dX)))[-1]
    worstLC=max(worstLC,abs(DL(T1)-DL(T2))/(Om*4*d))   # Lip of (2t-1)^2 on [0,1] operator = 4 ? (check)
print(worstQ,worstH,worstLC)
