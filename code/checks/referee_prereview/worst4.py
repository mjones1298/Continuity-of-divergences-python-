from pathlib import Path
import numpy as np; from lib import *; from scipy.optimize import minimize
exec(open(Path(__file__).resolve().parent / 'worst3.py').read().split('for p in')[0].split('rng=')[0])
rng=np.random.default_rng(4)
def rx(t): return np.array([[np.cos(t/2),-1j*np.sin(t/2)],[-1j*np.sin(t/2),np.cos(t/2)]])
Pa=[np.eye(2),np.array([[0,1],[1,0]]),np.array([[0,-1j],[1j,0]]),np.diag([1,-1])]
def tn(A): return np.sum(np.abs(np.linalg.eigvalsh(A)))
def Kd(p,dl): return [np.sqrt(1-3*p/4)*rx(np.pi/2+dl)]+[np.sqrt(p/4)*s@rx(np.pi/2+dl) for s in Pa[1:]]
def pert(K,x,eps):
    X=(x[:16]+1j*x[16:]).reshape(4,2,2); X=X/np.linalg.norm(X)*eps
    Kp=[k+x_ for k,x_ in zip(K,X)]; S=sum(k.conj().T@k for k in Kp); R=hp(S,-.5); return choi([k@R for k in Kp])
print("Interior pair: Phi=gate+dep p, Psi=gate rotated by dl0 with F~0.5; perturb Psi; worst |dF| vs Choi distance d")
for p in [1e-2,1e-3,1e-4]:
    # find dl0 with F ~ 0.5
    J1=choi(Kd(p,0)); dls=np.logspace(-4,0,400); dl0=[d for d in dls if geo(J1,choi(Kd(p,d)))<0.5][0]
    K=Kd(p,dl0); J2=choi(K); F0=geo(J1,J2); out=[]
    for scale in [0.01,0.03,0.1]:
        eps=scale*np.sqrt(p); best=(0,0)
        for sgn in [1,-1]:
          for _ in range(3):
            f=lambda x: -sgn*(geo(J1,pert(K,x,eps))-F0)
            r=minimize(f,rng.normal(size=32),method='Powell',options={'maxiter':2500})
            J=pert(K,r.x,eps); v=abs(geo(J1,J)-F0); d=tn(J-J2)
            if v>best[0]: best=(v,d)
        out.append(best)
    print(f"p={p:.0e} dl0={dl0:.3f} F0={F0:.2f}: "+" | ".join(f"d={d:.1e}: |dF|={v:.4f}, |dF|sqrt(p)/d={v*np.sqrt(p)/d:.2f}, |dF|/sqrt(d/p)={v/np.sqrt(d/p):.3f}" for v,d in out),flush=True)
