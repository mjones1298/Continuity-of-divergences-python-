import numpy as np, scipy.integrate as si, warnings
warnings.filterwarnings("ignore")
rng=np.random.default_rng(5)
def kap(s,t):
    return (2*t-1)**2/(s*(1-t)+t) if not np.isinf(s) else (2*t-1)**2/(1-t)
def thm98(I,a,b,c,dens,atoms,delta):
    lo,hi=I
    ts=np.linspace(lo,hi,4001)
    rho=np.max(abs(2*ts-1))
    def term(s):
        if np.isinf(s): return None
        d=(s*(1-ts)+ts).min()
        lip=rho*(2/d+(1+s)/d**2)
        k=kap(s,ts); osc=k.max()-k.min()
        return min(lip*delta,osc)
    tot=abs(2*b-a)*delta
    if c>0:
        d=(1-ts).min(); tot+=c*rho*(2/d+1/d**2)*delta
    if dens is not None:
        f=lambda s: term(s)*dens(s)
        tot+=si.quad(f,0,1,limit=400)[0]+si.quad(f,1,50,limit=400)[0]+si.quad(f,50,np.inf,limit=400)[0]
    for s0,w in atoms: tot+=w*term(s0)
    return tot
def true_mod(h,I,delta,ntr=300):
    lo,hi=I; best=0
    # scalar
    ts=np.linspace(lo,hi,2001)
    for dd in [delta]:
        for t in ts:
            for u in (t+dd,t-dd):
                if lo<=u<=hi: best=max(best,abs(h(t)-h(u)))
    # matrices 2x2 random + structured
    for _ in range(ntr):
        n=rng.integers(2,4)
        Q=np.linalg.qr(rng.normal(size=(n,n))+1j*rng.normal(size=(n,n)))[0]
        X=Q@np.diag(rng.uniform(lo,hi,n))@Q.conj().T
        H=rng.normal(size=(n,n))+1j*rng.normal(size=(n,n)); H=H+H.conj().T; H*=dd/np.linalg.norm(H,2)
        Y=X+H
        w,V=np.linalg.eigh(Y)
        if w.min()<lo or w.max()>hi: continue
        fX=lambda A:(lambda w,V:(V*h(w))@V.conj().T)(*np.linalg.eigh(A))
        best=max(best,np.linalg.norm(fX(X)-fX(Y),2))
    return best
pi=np.pi
cases={
 'Hellinger[0,1]':( (0,1),0,0,0,lambda s:2/pi*np.sqrt(s)/(1+s)**2,[],lambda t:1-2*np.sqrt(t*(1-t))),
 'JS[0,1]':( (0,1),0,0,0,lambda s:min(s,1)/(1+s)**2,[],lambda t:np.log(2)+np.where(t>0,t*np.log(np.maximum(t,1e-300)),0)+np.where(t<1,(1-t)*np.log(np.maximum(1-t,1e-300)),0)),
 'LeCam[0,1]':((0,1),0,0,0,None,[(1.0,1.0)],lambda t:(2*t-1)**2),
 'BS[0,.8]':((0,0.8),0,1,0,lambda s:s/(1+s)**2,[],lambda t:np.where(t>0,t*np.log(np.maximum(t,1e-300)/(1-t)),0)),
 'Pearson[0,.8]':((0,0.8),0,0,1,None,[],lambda t:(2*t-1)**2/(1-t)),
}
for name,(I,a,b,c,dens,atoms,h) in cases.items():
    for delta in [0.3,0.1,0.01,0.001]:
        B=thm98(I,a,b,c,dens,atoms,delta)
        T=true_mod(h,I,delta)
        flag="" if B>=T*(1-1e-9) else "  <<<< BOUND VIOLATED"
        print("%-15s delta=%-6g bound=%.6g  true>=%.6g%s"%(name,delta,B,T,flag))
