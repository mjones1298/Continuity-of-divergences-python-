import numpy as np, mpmath as mp
from scipy.optimize import minimize
from scipy.linalg import logm, sqrtm
from qlib import *
mp.mp.dps=40
def JNn(p,th):
    Rz=np.diag([np.exp(-1j*th),np.exp(1j*th)]); v=vec(Rz@H)
    return (1-p)*np.outer(v,v.conj())+p/2*np.eye(4)
def JNm(p,th):
    p=mp.mpf(p);t=mp.mpf(th);Hm=mp.matrix([[1,1],[1,-1]])/mp.sqrt(2)
    V=mp.matrix([[mp.exp(-1j*t),0],[0,mp.exp(1j*t)]])*Hm
    return (1-p)*mpouter([V[0,0],V[0,1],V[1,0],V[1,1]])+p/2*mp.eye(4)
def rho_of(x):
    r=np.array(x); n=np.linalg.norm(r)
    if n>1: r=r/n
    return (I2+r[0]*X+r[1]*Y+r[2]*Z)/2
def relent_mp(A,B):
    wa,Va=mp.eigh(A); wb,Vb=mp.eigh(B)
    def lg(w,V):
        D=mp.zeros(len(w),len(w))
        for i in range(len(w)): D[i,i]=mp.log(w[i]) if w[i]>0 else mp.mpf(0)
        return V*D*V.H
    return mp.re(sum((A*(lg(wa,Va)-lg(wb,Vb)))[i,i] for i in range(A.rows)))
def D_in(p,th,rho):
    s=mp.matrix(sqrtm(rho).T.tolist()); S=mpkron(mp.eye(2),s)
    return relent_mp(S*JNm(p,0)*S, S*JNm(p,th)*S)
def Dn(p,th,rho):
    s=sqrtm(rho).T; S=np.kron(I2,s)
    A=S@JNn(p,0)@S; B=S@JNn(p,th)@S
    wa,va=np.linalg.eigh(A); wb,vb=np.linalg.eigh(B)
    la=va@np.diag(np.log(np.clip(wa,1e-300,None)))@va.conj().T
    lb=vb@np.diag(np.log(np.clip(wb,1e-300,None)))@vb.conj().T
    return np.real(np.trace(A@(la-lb)))
rng=np.random.default_rng(1)
for p in [1e-2,1e-3,1e-4]:
    th=0.03*np.sqrt(p)
    best=(-1,None)
    for s in range(15):
        x0=rng.normal(size=3); x0=x0/np.linalg.norm(x0)*rng.uniform(0.1,0.99)
        r=minimize(lambda x:-Dn(p,th,rho_of(x)),x0,method='Nelder-Mead',options={'xatol':1e-8,'fatol':1e-20})
        if -r.fun>best[0]: best=(-r.fun,r.x)
    rho=rho_of(best[1])
    val=D_in(p,th,rho); val2=D_in(p,th/10,rho); valmax=D_in(p,th,I2/2)
    Fm=fid_mp(JNm(p,0),JNm(p,th/10),2,2,'0',dps=40)
    loc=(1-Fm)/(mp.mpf(th/10)**2)
    print(f"p={p:.0e}: opt Bloch={np.round(best[1],4)} |r|={np.linalg.norm(best[1]):.4f}; D/th^2 (th=0.03sqrt p)={mp.nstr(val/th**2,7)}, (th=0.003 sqrt p)={mp.nstr(val2/(th/10)**2,7)}; max-ent input D/th^2={mp.nstr(valmax/th**2,6)}")
    print(f"     D/th^2 / log(1/p) = {mp.nstr(val2/(th/10)**2/mp.log(1/p),6)} ; (1-F)/th^2 = {mp.nstr(loc,7)} ; times p = {mp.nstr(loc*p,6)}")
