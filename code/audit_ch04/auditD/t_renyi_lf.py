import numpy as np, warnings, scipy.integrate as si
from schur import schur_norm
from lib import Mdd
warnings.filterwarnings("ignore")
def Ms(s,eps):
    rho=1-2*eps
    d=eps*(1+s)+rho*min(1,s)
    return rho*(2/d+(1+s)/d**2)
def Lf(a,b,dens,eps):
    f=lambda s:Ms(s,eps)*dens(s)
    return abs(2*b-a)+si.quad(f,0,1,limit=300)[0]+si.quad(f,1,100,limit=300)[0]+si.quad(f,100,np.inf,limit=300)[0]
pi=np.pi
rows=[]
for al in [1.2,1.5,1.9]:
    dens=lambda s,al=al:np.sin(pi*(al-1))/pi*s**al/(1+s)**2
    g=lambda t,al=al:t**al*(1-t)**(1-al); gp=lambda t,al=al:t**(al-1)*(1-t)**(-al)*(al-t)
    for eps in [0.3,0.1,0.02]:
        lo,hi=eps,1-eps
        lip=max(schur_norm(Mdd(g,gp,np.linspace(lo,hi,n))) for n in (2,5,9))
        L=Lf(1,al,dens,eps)
        print("Renyi a=%.1f eps=%.2f Lip~%.5f h'(1-eps)=%.5f L_f=%.5f  L_f/Lip=%.3f"%(al,eps,lip,gp(hi),L,L/lip))
al=-0.5
dens=lambda s:-np.sin(pi*al)/pi*s**al/(1+s)**2
g=lambda t:t**al*(1-t)**(1-al); gp=lambda t:t**(al-1)*(1-t)**(-al)*(al-t)
for eps in [0.3,0.1,0.02]:
    lo,hi=eps,1-eps
    lip=max(schur_norm(Mdd(g,gp,np.linspace(lo,hi,n))) for n in (2,5,9))
    L=Lf(1,al,dens,eps)
    print("Renyi a=%.1f eps=%.2f Lip~%.5f |h'(eps)|=%.5f L_f=%.5f  L_f/Lip=%.3f ; table claim (eps-al)(1-eps)^-al/eps^(1-al)=%.5f"%(al,eps,lip,abs(gp(lo)),L,L/lip,(eps-al)*(1-eps)**(-al)/eps**(1-al)))
dens=lambda s:s/(1+s)**2
gp=lambda t:np.log(t/(1-t))+1/(1-t); g=lambda t:t*np.log(t/(1-t))
for eps in [0.3,0.1,0.02]:
    lo,hi=eps,1-eps
    lip=max(schur_norm(Mdd(g,gp,np.linspace(lo,hi,n))) for n in (2,5,9))
    L=Lf(0,1,dens,eps)
    print("BS eps=%.2f Lip~%.5f h'(1-eps)=%.5f L_f=%.5f L_f/Lip=%.3f 3N=%.3f"%(eps,lip,gp(hi),L,L/lip,3*max(abs(gp(lo)),abs(gp(hi)))))
