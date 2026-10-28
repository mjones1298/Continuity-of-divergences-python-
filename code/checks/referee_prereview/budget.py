import numpy as np, itertools; from lib import *
rng=np.random.default_rng(3)
I=np.eye(2);X=np.array([[0,1],[1,0]]);Y=np.array([[0,-1j],[1j,0]]);Z=np.diag([1,-1]).astype(complex)
P={'I':I,'X':X,'Y':Y,'Z':Z}
def eb(s): w,V=np.linalg.eigh(P[s]); return V[:,::-1]
def tomo(rho,N):
    ex={}
    for a,b in itertools.product('XYZ','XYZ'):
        U=np.kron(eb(a),eb(b)); pr=np.clip(np.real(np.diag(U.conj().T@rho@U)),0,None); pr/=pr.sum()
        f=rng.multinomial(N,pr)/N; sa=np.array([1,1,-1,-1]); sb=np.array([1,-1,1,-1])
        for (pa,pb,s) in [(a,b,sa*sb),(a,'I',sa),('I',b,sb)]: ex.setdefault(pa+pb,[]).append(f@s)
    R=np.eye(4)/4
    for k,v in ex.items(): R=R+np.mean(v)*np.kron(P[k[0]],P[k[1]])/4
    w,V=np.linalg.eigh(R); w=np.clip(w,0,None); w/=w.sum(); return (V*w)@V.conj().T
def ad(g): return [np.array([[1,0],[0,np.sqrt(1-g)]]),np.array([[0,np.sqrt(g)],[0,0]])]
def hs(J1,J2,g): # product-input lower bound proxy not needed; use Choi-state hockey stick
    w=np.linalg.eigvalsh(J1/2-g*J2/2); return w[w>0].sum()
print("shots/setting needed for 90% of runs within 10% of true value (AD .90 vs .99, depol floor p)")
for p in [1e-2,1e-3,1e-4]:
    JN=dep(choi(ad(.9)),p); JM=dep(choi(ad(.99)),p)
    tg=-2*np.log(geo(JN,JM)); th=hs(JN,JM,2.0); tu=uhlF(JN/2,JM/2)
    res={}
    for N in [10**k for k in range(3,11)]:
        g=[];h=[];u=[]
        for _ in range(60):
            a=tomo(JN/2,N);b=tomo(JM/2,N)
            g.append(-2*np.log(max(geo(2*a,2*b),1e-300))); h.append(hs(2*a,2*b,2.0)); u.append(uhlF(a,b))
        ok=lambda v,t: np.mean(np.abs(np.array(v)-t)<=0.1*abs(t))>=0.9
        for name,v,t in [('geo',g,tg),('hs',h,th),('uhl',u,tu)]:
            if name not in res and ok(v,t): res[name]=N
    print(f"p={p:.0e} true geo={tg:.2f}: N_geo={res.get('geo','>1e10')} N_hockey={res.get('hs','>1e10')} N_uhl={res.get('uhl','>1e10')}")
