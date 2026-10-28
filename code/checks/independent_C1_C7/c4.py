import numpy as np, mpmath as mp
from qlib import *
from scipy.optimize import minimize
mp.mp.dps=50
Hm=mp.matrix([[1,1],[1,-1]])/mp.sqrt(2)
def mpvec(K): return [K[0,0],K[0,1],K[1,0],K[1,1]]
def JN(p,th,half=False):
    p=mp.mpf(p); t=mp.mpf(th)/(2 if half else 1)
    Rz=mp.matrix([[mp.exp(-1j*t),0],[0,mp.exp(1j*t)]])
    V=Rz*Hm
    return (1-p)*mpouter(mpvec(V))+p/2*mp.eye(4)
def F(p,th,half=False): return fid_mp(JN(p,0),JN(p,th,half),2,2,'0',dps=50)
for p in ['1e-2','1e-3','1e-4']:
    P_=mp.mpf(p)
    for half in [False,True]:
        th0=mp.findroot(lambda t: F(p,t*mp.sqrt(P_),half)-mp.mpf('0.9'), 0.5 if not half else 1.0)
        # local constant c: F ~ (1+c th^2/p)^{-1/2} fit at several theta
        cs=[]
        for k in ['0.1','0.5','1','2']:
            t=mp.mpf(k)*mp.sqrt(P_); f=F(p,t,half)
            cs.append((1/f**2-1)*P_/t**2)
        print(f"p={p} conv={'e^{-i th Z/2}' if half else 'e^{-i th Z}'}: theta(F=0.9)/sqrt(p)={mp.nstr(th0,6)}; c=(F^-2-1)p/th^2 at th/sqrt p=0.1,0.5,1,2: {[mp.nstr(c,6) for c in cs]}")
    # compare to formula at a few theta
    for k in ['0.5','1']:
        t=mp.mpf(k)*mp.sqrt(P_)
        print("   th/sqrt p=",k,"F=",mp.nstr(F(p,t),8)," (1+th^2/p)^-1/2=",mp.nstr((1+t**2/P_)**-0.5,8))
print("ideal theta for 0.9 with c=1:", np.sqrt(1/0.81-1))
# SDP cross-check one point
s=fid_sdp(mp2np(JN('1e-2',0)),mp2np(JN('1e-2',0.05)),2,2); print('SDP p=1e-2 th=0.05:',s,' mp:',mp.nstr(F('1e-2',0.05),10))
# nu^2
def nu2(Jom, starts=200, seed=0):
    w,Vv=np.linalg.eigh(Jom); idx=w>1e-14*w.max()
    Ks=[np.sqrt(w[k])*Vv[:,k].reshape(2,2) for k in np.where(idx)[0]]
    n=len(Ks); rng=np.random.default_rng(seed)
    def f(x):
        c=x[:n]+1j*x[n:]; c=c/np.linalg.norm(c)
        return np.linalg.norm(sum(ci*K for ci,K in zip(c,Ks)),2)**2
    best=np.inf
    for s in range(starts):
        r=minimize(f,rng.normal(size=2*n),method='Nelder-Mead',options={'xatol':1e-12,'fatol':1e-16,'maxiter':20000})
        best=min(best,r.fun)
    return best
for p in [1e-2,1e-3,1e-4]:
    J0=mp2np(JN(p,0)); 
    out=[nu2(2*J0,starts=60)]
    for th in [1e-2*np.sqrt(p),0.1*np.sqrt(p),np.sqrt(p)]:
        out.append(nu2(J0+mp2np(JN(p,th)),starts=60))
    print(f"p={p}: nu^2(2N_p)={out[0]:.6e} (p/2={p/2:.1e}); nu^2(N_p+N_p,th) at th/sqrt p=0.01,0.1,1: {[f'{x:.4e}' for x in out[1:]]}")
