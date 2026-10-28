import numpy as np; from lib import *; from scipy.optimize import minimize
rng=np.random.default_rng(9)
def rx(t): return np.array([[np.cos(t/2),-1j*np.sin(t/2)],[-1j*np.sin(t/2),np.cos(t/2)]])
Pa=[np.eye(2),np.array([[0,1],[1,0]]),np.array([[0,-1j],[1j,0]]),np.diag([1,-1])]
def Kd(p,dl): return [np.sqrt(1-3*p/4)*rx(np.pi/2+dl)]+[np.sqrt(p/4)*s@rx(np.pi/2+dl) for s in Pa[1:]]
def tn(A): return np.sum(np.abs(np.linalg.eigvalsh(A)))
def hlog(M): w,V=np.linalg.eigh((M+M.conj().T)/2); return (V*np.log(np.clip(w,1e-300,None)))@V.conj().T
def um(A,B): return np.real(np.trace(A@(hlog(A)-hlog(B))))
def bs(A,B): Ah=hp(A,.5); return np.real(np.trace(Ah@hlog(Ah@hp(B,-1)@Ah)@Ah))
def gl(J1,J2): return -2*np.log(geo(J1,J2))
def pert(K,x,eps):
    X=(x[:16]+1j*x[16:]).reshape(4,2,2); X=X/np.linalg.norm(X)*eps
    Kp=[k+x_ for k,x_ in zip(K,X)]; S=sum(k.conj().T@k for k in Kp); R=hp(S,-.5); return choi([k@R for k in Kp])
dl0=0.05
print(f"Pair: Phi=gate(+dl0={dl0})+dep p, Psi=gate+dep p. Perturb FIRST argument Phi (Kraus-level, TP kept). Worst |Delta|/d, d=Choi trace-norm dist (d≈1e-4 fixed)")
print("p      -2logF_geo   D_BS(Choi)   D_Um(Choi)   [values]")
for p in [1e-2,1e-3,1e-4]:
    K=Kd(p,dl0); J1=choi(K); J2=choi(Kd(p,0))
    fs={'geo':lambda A:gl(A,J2),'bs':lambda A:bs(A/2,J2/2),'um':lambda A:um(A/2,J2/2)}
    v0={k:f(J1) for k,f in fs.items()}; res={}
    eps=1e-4
    for k,f in fs.items():
        best=0
        for _ in range(4):
            for sg in [1,-1]:
                r=minimize(lambda x:-sg*(f(pert(K,x,eps))-v0[k]),rng.normal(size=32),method='Powell',options={'maxiter':2000})
                J=pert(K,r.x,eps); best=max(best,abs(f(J)-v0[k])/tn(J-J1))
        res[k]=best
    print(f"{p:.0e}  {res['geo']:10.1f}  {res['bs']:10.1f}  {res['um']:10.2f}   [{v0['geo']:.3f}, {v0['bs']:.3f}, {v0['um']:.4f}]",flush=True)
