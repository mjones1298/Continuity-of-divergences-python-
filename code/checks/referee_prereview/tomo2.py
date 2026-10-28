from pathlib import Path
import numpy as np
exec(open(Path(__file__).resolve().parent / 'tomo.py').read().split('gN,gM=0.90,0.99')[0])
def gm(A,B,dl=1e-14):
    A=A+dl*np.eye(4);B=B+dl*np.eye(4)
    def hp(M,a):
        w,V=np.linalg.eigh((M+M.conj().T)/2); w=np.clip(w,1e-30,None); return (V*w**a)@V.conj().T
    Ah=hp(A,.5);Aih=hp(A,-.5); return Ah@hp(Aih@B@Aih,.5)@Ah
def geo(J1,J2): return np.min(np.linalg.eigvalsh(ptrB(gm(J1,J2))))
def rank(J,t=1e-12): return np.sum(np.linalg.eigvalsh(J)>t)
def bs(J1,J2):
    if rank(J2)<4: 
        # support check
        w,V=np.linalg.eigh(J2); P0=V[:,w<=1e-12]
        if np.linalg.norm(P0.conj().T@J1@P0)>1e-12: return np.inf
    return bs_chan(J1,J2)
gN,gM=0.90,0.99
for p in [1e-2,1e-3]:
    JN=dep(choi(ad(gN)),p); JM=dep(choi(ad(gM)),p)
    tg=-2*np.log(geo(JN,JM)); tb=bs_chan(JN,JM)
    print(f"p={p:.0e} TRUE geo={tg:.3f} BS={tb:.3f}")
    for N in [10**3,10**4,10**5,10**6]:
        g=[];b=[];rk=[]
        for _ in range(200):
            rn=2*tomo(JN/2,N); rm=2*tomo(JM/2,N)
            f=geo(rn,rm); g.append(-2*np.log(max(f,1e-300))); b.append(bs(rn,rm)); rk.append(rank(rm)<4)
        g=np.array(g);b=np.array(b)
        q=lambda a:np.percentile(a,[5,50,95])
        fb=np.isfinite(b)
        print(f"  N={N:.0e}: geo med {q(g)[1]:.2f} frac(geo>2*true) {np.mean(g>2*tg):.2f} [5%:{q(g)[0]:.2f}, 95%:{q(g)[2]:.2f}] | BS infinite in {100*(1-fb.mean()):.0f}% ; finite med {np.median(b[fb]) if fb.any() else np.nan:.2f} | est. J_M rank-deficient {100*np.mean(rk):.0f}%")
