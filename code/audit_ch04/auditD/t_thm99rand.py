import numpy as np, warnings, sys
from schur import schur_norm
from lib import Mdd, kappa
warnings.filterwarnings("ignore")
rng=np.random.default_rng(99)

def Ms(s,eps):
    rho=1-2*eps
    if np.isinf(s): return (1-4*eps**2)/eps**2
    d=eps*(1+s)+rho*min(1,s)
    return rho*(2/d+(1+s)/d**2)

def build(a,b,c,atoms):
    ks=[(c,kappa(np.inf))]+[(w,kappa(s)) for (s,w) in atoms]
    h=lambda t:(a*(1-t)+b*(2*t-1))+sum(w*k(t) for w,(k,kp) in ks)
    hp=lambda t:(2*b-a)+sum(w*kp(t) for w,(k,kp) in ks)
    return h,hp
def Lf(a,b,c,atoms,eps):
    return abs(2*b-a)+c*Ms(np.inf,eps)+sum(w*Ms(s,eps) for s,w in atoms)

stat={'a_ub':0,'a_3':0,'b1':0,'b2':0,'c':0,'n':0}
for trial in range(40):
    eps=rng.choice([0.3,0.1,0.03]); lo,hi=eps,1-eps
    mode=rng.choice(['gen','hi','lo','sym'])
    nat=rng.integers(1,4)
    if mode=='hi':
        atoms=[(rng.choice([1.0,rng.uniform(1,6),rng.uniform(6,60)]),rng.uniform(0,3)) for _ in range(nat)]
        c=rng.uniform(0,2)*rng.integers(0,2); a=rng.normal(); b=(a+abs(rng.normal())*rng.uniform(0,3))/2  # 2b>=a
    elif mode=='lo':
        atoms=[(rng.choice([0.0,rng.uniform(0,1),1.0]),rng.uniform(0,3)) for _ in range(nat)]
        c=0.0; a=rng.normal(); b=(a-abs(rng.normal())*rng.uniform(0,3))/2  # 2b<=a
    elif mode=='gen':
        atoms=[(rng.choice([0.0,rng.uniform(0,1),rng.uniform(1,6),rng.uniform(6,60)]),rng.uniform(0,3)) for _ in range(nat)]
        c=rng.uniform(0,2)*rng.integers(0,2); a=rng.normal(); b=rng.normal()
    else:
        # symmetric: f(x)=x f(1/x): use sym pairs s and 1/s? h_f symmetric iff mu symmetric under s->1/s (with weight) and c=0 ... construct via kappa sym directly
        atoms=[]; c=0
    if mode=='sym':
        ks=[(rng.uniform(0,3),rng.choice([0.0,rng.uniform(0,1),1.0,rng.uniform(1,6),np.inf])) for _ in range(rng.integers(1,4))]
        def hsym(t,ks=ks):
            return sum(w*(kappa(s)[0](t)+kappa(s)[0](1-t)) for w,s in ks)
        def hpsym(t,ks=ks):
            return sum(w*(kappa(s)[1](t)-kappa(s)[1](1-t)) for w,s in ks)
        h,hp=hsym,hpsym
        Lipclaim=hp(hi)
        Lfv=None
    else:
        h,hp=build(a,b,c,atoms); Lfv=Lf(a,b,c,atoms,eps)
    ts=np.linspace(lo,hi,2001); N=max(abs(hp(ts)))
    ends=schur_norm(Mdd(h,hp,np.array([lo,hi])))
    grid=max(schur_norm(Mdd(h,hp,np.linspace(lo,hi,n))) for n in (4,7,11))
    stat['n']+=1
    best=max(ends,grid)
    if Lfv is not None:
        if best>Lfv*(1+1e-7): stat['a_ub']+=1; print("Lip>L_f!",mode,best,Lfv)
        if Lfv>3*N*(1+1e-7): stat['a_3']+=1; print("L_f>3N!",mode,Lfv,N)
        if mode=='hi' and abs(Lfv-hp(hi))>1e-7*Lfv: stat['b1']+=1; print("L_f != h'(t+) hi mode",Lfv,hp(hi))
        if mode=='hi' and abs(best-Lfv)>1e-6*Lfv: stat['b1']+=1; print("Lip!=L_f (b first case)",best,Lfv)
        if mode=='lo' and abs(best-Lfv)>1e-6*Lfv: stat['b2']+=1; print("Lip!=L_f (b second case)",best,Lfv, -hp(lo))
        print("%-4s eps=%.2f Lip~%.5f ends=%.5f L_f=%.5f 3N=%.5f"%(mode,eps,best,ends,Lfv,3*N)); 
    else:
        if abs(best-Lipclaim)>1e-6*Lipclaim: stat['c']+=1; print("(c) FAIL",best,Lipclaim)
        print("sym  eps=%.2f Lip~%.5f claim h'(t+)=%.5f"%(eps,best,Lipclaim))
    sys.stdout.flush()
print(stat)
