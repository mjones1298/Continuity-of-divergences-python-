from pathlib import Path
import numpy as np, itertools
rng=np.random.default_rng(7)
exec(open(Path(__file__).resolve().parent / 'mc.py').read().split('def umeg_chan')[0])
I=np.eye(2);X=np.array([[0,1],[1,0]]);Y=np.array([[0,-1j],[1j,0]]);Z=np.diag([1,-1])
P={'I':I,'X':X,'Y':Y,'Z':Z}
def eigbasis(s):
    w,V=np.linalg.eigh(P[s]); return V[:,::-1]  # columns: +1, -1
def tomo(rho,N):
    # 9 product Pauli settings, N shots each; linear inversion; clip to PSD
    ex={}
    cnt={}
    for a,b in itertools.product('XYZ','XYZ'):
        U=np.kron(eigbasis(a),eigbasis(b)); pr=np.real(np.diag(U.conj().T@rho@U)); pr=np.clip(pr,0,None); pr/=pr.sum()
        f=rng.multinomial(N,pr)/N; sgn={'a':np.array([1,1,-1,-1]),'b':np.array([1,-1,1,-1])}
        for (pa,pb,s) in [(a,b,sgn['a']*sgn['b']),(a,'I',sgn['a']),('I',b,sgn['b'])]:
            ex.setdefault(pa+pb,[]).append(f@s)
    R=np.eye(4)/4
    for k,v in ex.items(): R=R+np.mean(v)*np.kron(P[k[0]],P[k[1]])/4
    w,V=np.linalg.eigh(R); w=np.clip(w,0,None); w/=w.sum(); return (V*w)@V.conj().T
gN,gM=0.90,0.99
print("Shots per setting N (9 settings/channel); estimated -log F_geo^2 and D_BS (mean, 5-95%); TRUE value; plus Choi-state Uhlmann infidelity baseline")
for p in [1e-2,1e-3]:
    JN=dep(choi(ad(gN)),p); JM=dep(choi(ad(gM)),p)
    tg=-2*np.log(geo_chan_fid(JN,JM)); tb=bs_chan(JN,JM)
    def uhl(A,B):
        sA=hpow(A,.5); return np.sum(np.sqrt(np.clip(np.linalg.eigvalsh(sA@B@sA),0,None)))
    tu=1-uhl(JN/2,JM/2)**2
    print(f"p={p:.0e} TRUE geo={tg:.3f} BS={tb:.3f} Choi-Uhlmann-infid={tu:.4f}")
    for N in [10**3,10**4,10**5,10**6]:
        g=[];b=[];u=[];inf=0
        for _ in range(100):
            rn=tomo(JN/2,N); rm=tomo(JM/2,N)
            f=geo_chan_fid(2*rn,2*rm)
            if f<=1e-12: inf+=1; continue
            g.append(-2*np.log(f)); b.append(bs_chan(2*rn,2*rm)); u.append(1-uhl(rn,rm)**2)
        q=lambda a:np.percentile(a,[5,50,95])
        print(f"  N={N:.0e}: geo med {q(g)[1]:.3f} [{q(g)[0]:.3f},{q(g)[2]:.3f}] infinite:{inf}/100 | BS med {q(b)[1]:.3f} [{q(b)[0]:.3f},{q(b)[2]:.3f}] | Uhl {q(u)[1]:.4f} [{q(u)[0]:.4f},{q(u)[2]:.4f}]")
