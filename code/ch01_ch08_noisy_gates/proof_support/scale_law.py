"""Scale-resolved comparison at qubit depolarized gates N_{p,delta} = D_p o R_X(delta) o U vs N_p = D_p o U.
Closed forms (verified in chk_qubit_closed.py):
  x = 4(1-p)^2 sin^2(delta/2) / (p(4-3p))
  Fhat = p/2 + (1-p/2)/sqrt(1+x);  D_BS(channel) = (2-p) sqrt(x/(1+x)) asinh(sqrt x)
  D(Choi states) = (1-p) sin^2(delta/2) log(1 + 4(1-p)/p)
Stein exponent zeta bounds:
  lower: max_k (1/k) D(J_k/2 || J_k'/2), k sequential uses (D_{p_k} o R(k delta)), p_k = 1-(1-p)^k
  upper: min{ D_BS (Fang-Fawzi chain rule), paper-3 Cor stein(b) bound with Phi'=Psi }.
Chernoff: Chat = -ln min_s tr(J2 #_s J1)/2 (Tr_Y prop. to identity), xi lower: max_k (1/k) C_QC(block Choi states)."""
import numpy as np
from scipy.optimize import minimize_scalar
def xval(p,d): return 4*(1-p)**2*np.sin(d/2)**2/(p*(4-3*p))
def Fhat(p,d): x=xval(p,d); return p/2+(1-p/2)/np.sqrt(1+x)
def DBS(p,d): x=xval(p,d); return (2-p)*np.sqrt(x/(1+x))*np.arcsinh(np.sqrt(x))
def Dchoi(p,d): return (1-p)*np.sin(d/2)**2*np.log(1+4*(1-p)/p)
def zeta_lower(p,d,kmax=None):
    best=(Dchoi(p,d),1)
    kmax = kmax or int(min(50/p, 50/ max(d,1e-12)))+2
    ks=np.unique(np.round(np.geomspace(1,kmax,400)).astype(int))
    for k in ks:
        pk=1-(1-p)**k; dk=k*d
        if dk>np.pi: break
        v=Dchoi(pk,dk)/k
        if v>best[0]: best=(v,k)
    return best
def hbin(q): return 0 if q<=0 else -q*np.log(q)-(1-q)*np.log(1-q)
def zeta_upper_p3(p,d):
    e=(1-p)*np.sin(d/2); kappa=(2-1.5*p)/(p/2)
    return 2*e*np.log(2)+(1+e)*hbin(e/(1+e))+e*np.log(kappa)
def qchernoff(q,c):
    # rho=(1-q)|a><a|+q/4, sigma=(1-q)|b><b|+q/4 in C^4, |<a,b>|=c ; restrict to span{a,b} + complement
    lam=1-3*q/4; mu=q/4
    a=np.array([1,0.]); b=np.array([c,np.sqrt(1-c*c)])
    def Q(s):
        # on span{a,b}: 2x2; complement (2 dims): mu
        A=mu*np.eye(2)+(lam-mu)*np.outer(a,a); B=mu*np.eye(2)+(lam-mu)*np.outer(b,b)
        wa,Ua=np.linalg.eigh(A); wb,Ub=np.linalg.eigh(B)
        As=(Ua*wa**s)@Ua.T; Bs=(Ub*wb**(1-s))@Ub.T
        return np.trace(As@Bs)+2*mu
    r=minimize_scalar(Q,bounds=(0,1),method='bounded'); return -np.log(r.fun)
def xi_lower(p,d):
    best=0
    kmax=int(min(50/p,50/max(d,1e-12)))+2
    for k in np.unique(np.round(np.geomspace(1,kmax,200)).astype(int)):
        pk=1-(1-p)**k; dk=k*d
        if dk>np.pi: break
        best=max(best,qchernoff(pk,np.cos(dk/2))/k)
    return best
def Chat(p,d):
    # tr(J2 #_s J1)/2 via 2x2 blocks: J=(1-p)2|a><a| + p/2 ; tr over complement: 2*(p/2)
    lam=2-1.5*p; mu=p/2; c=np.cos(d/2)
    a=np.array([1,0.]); b=np.array([c,np.sqrt(1-c*c)])
    A=mu*np.eye(2)+(lam-mu)*np.outer(a,a); B=mu*np.eye(2)+(lam-mu)*np.outer(b,b)
    def gm(s):
        w,U=np.linalg.eigh(B); Bh=(U*np.sqrt(w))@U.T; Bih=(U/np.sqrt(w))@U.T
        w2,U2=np.linalg.eigh(Bih@A@Bih); M=(U2*w2**s)@U2.T
        return (np.trace(Bh@M@Bh)+2*mu)/2
    r=minimize_scalar(gm,bounds=(0,1),method='bounded'); return -np.log(r.fun)
if __name__=="__main__":
    print("max_u f(u), f(u)=u e^{-u} log((1+3e^{-u})/(1-e^{-u})):")
    f=lambda u: u*np.exp(-u)*np.log((1+3*np.exp(-u))/(1-np.exp(-u)))
    r=minimize_scalar(lambda u:-f(u),bounds=(1e-6,10),method='bounded'); print("  u*=%.5f f*=%.5f  => zeta >= f*/4 * delta^2/p, BS/zeta <= 2/f* = %.4f"%(r.x,-r.fun,2/(-r.fun)))
    g=lambda c: np.sin(c/2)**2/c; r2=minimize_scalar(lambda c:-g(c),bounds=(0.1,3),method='bounded'); print("  max sin^2(c/2)/c = %.5f at c=%.4f"%(-r2.fun,r2.x))
    print("\n p       delta/p    x      BS      zeta_low (k)     zeta_up(p3)  BS/zeta_up  BS/zeta_low | Chat   xi_low  Chat/xi_low")
    for p in [1e-2,1e-3,1e-4]:
        for r_ in [0.01,0.1,1,3,10,30,100,300,1000]:
            d=r_*p
            if d>2: continue
            zl,k=zeta_lower(p,d); zu=min(DBS(p,d),zeta_upper_p3(p,d)); zu3=zeta_upper_p3(p,d)
            C=Chat(p,d); xl=xi_lower(p,d)
            print(f"{p:6.0e} {r_:8.2f} {xval(p,d):8.3g} {DBS(p,d):8.3g} {zl:9.3g} ({k:5d}) {zu3:9.3g} {DBS(p,d)/zu:9.3g} {DBS(p,d)/zl:9.3g} | {C:8.3g} {xl:8.3g} {C/xl:8.3g}")
