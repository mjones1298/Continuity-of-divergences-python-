import numpy as np; from lib import *; from scipy.optimize import minimize
rng=np.random.default_rng(2)
def rx(t): return np.array([[np.cos(t/2),-1j*np.sin(t/2)],[-1j*np.sin(t/2),np.cos(t/2)]])
Pa=[np.eye(2),np.array([[0,1],[1,0]]),np.array([[0,-1j],[1j,0]]),np.diag([1,-1])]
def tn(A): return np.sum(np.abs(np.linalg.eigvalsh(A)))
def K0(p): return [np.sqrt(1-3*p/4)*rx(np.pi/2)]+[np.sqrt(p/4)*s@rx(np.pi/2) for s in Pa[1:]]
def pert(K,x,eps):
    X=(x[:16]+1j*x[16:]).reshape(4,2,2); X=X/np.linalg.norm(X)*eps
    Kp=[k+x_ for k,x_ in zip(K,X)]; S=sum(k.conj().T@k for k in Kp); R=hp(S,-.5)
    return choi([k@R for k in Kp])
for p in [1e-2,1e-3,1e-4]:
    K=K0(p); J0=choi(K); out=[]
    for scale in [0.03,0.1,0.3,1.0]:
        eps=scale*np.sqrt(p); best=(0,0)
        for _ in range(5):
            f=lambda x: -(1-geo(J0,pert(K,x,eps)))
            r=minimize(f,rng.normal(size=32),method='Powell',options={'maxiter':3000})
            J=pert(K,r.x,eps); val=1-geo(J0,J); d=tn(J-J0)
            if val>best[0]: best=(val,d)
        out.append(best)
    print(f"p={p:.0e}: "+" | ".join(f"dChoi={d:.1e} (={d/np.sqrt(p):.2f}sqrt p): max1-F={v:.3f}, (1-F)p/d^2={v*p/d**2:.2f}" for v,d in out),flush=True)
