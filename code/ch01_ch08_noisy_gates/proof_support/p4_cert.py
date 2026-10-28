# P4: (1) Thompson-ball converse e^{-C(Phi,Psi)} >= (1-kappa) e^{-C(Phihat,Psihat)} on random ball members;
#     (2) two-point Pauli family numbers; (3) Pauli simulation KL bound; (4) coherent two-point Bhattacharyya.
import numpy as np, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))  # glib.py in code/common
from glib import *
X=np.array([[0,1],[1,0]],complex); Z=np.diag([1,-1]).astype(complex); Y=1j*X@Z
def Rx(d): return np.cos(d/2)*np.eye(2)-1j*np.sin(d/2)*X
def Ndep(p,d): return (1-p)*choi([Rx(d)])+p/2*np.eye(4)
def Chat(JPhi,JPsi):
    return -np.log(min(np.linalg.eigvalsh(trY(gmean_pd(JPsi,JPhi,s),2,2)).min() for s in np.linspace(0.01,0.99,99)))
rng=np.random.default_rng(11)
worst=-1
for trial in range(200):
    p=10**rng.uniform(-3,-1); d=rng.uniform(0,1)
    Ph=Ndep(p,d); Ps=Ndep(p,0)
    lam=min(np.linalg.eigvalsh(Ph).min(),np.linalg.eigvalsh(Ps).min())
    # random TP perturbations of diamond size eps
    def pert(J,eps):
        K=rand_kraus(rng,2,2,4); D=choi(K)-J
        dn=dnorm(D,2,2); t=min(1,eps/dn); return J+t*D, t*dn
    eps=rng.uniform(0.05,0.9)*lam
    Ph2,e1=pert(Ph,eps); Ps2,e2=pert(Ps,eps)
    kappa=2*max(e1,e2)/(2*lam)   # d_X*eps/(2 lam), d_X=2
    lhs=np.exp(-Chat(Ph2,Ps2)); rhs=(1-kappa)*np.exp(-Chat(Ph,Ps))
    worst=max(worst,(rhs-lhs)/rhs)
print('Thompson-ball Chernoff converse: max violation (should be <=0):',worst)
# two-point Pauli family: Phi0 = X (noiseless flip), Psi_q Pauli with q_X
for p in [1e-2,1e-3,1e-4]:
    lt=p/4; eta=0.05
    qX2=lt*((1+eta)/(1-eta))**2; s=qX2-lt
    q=np.array([1-3*p/4,p/4,p/4,p/4]); q2=q.copy(); q2[1]+=s; q2[0]-=s
    KL=np.sum(q*np.log(q/q2)); chi2=np.sum((q-q2)**2/q2)
    print(f'p={p}: lam~={lt:.1e} F ratio={np.sqrt(q2[1]/q[1]):.6f} diamond={2*s:.3e}=({2*s/(eta*lt):.2f})*eta*lam~ ; KL={KL:.3e}, KL/(eta^2 lam~)={KL/(eta**2*lt):.3f}; N_min>={np.log(2)/3/KL:.3e} = {np.log(2)/3/KL*eta**2*lt:.4f}/(eta^2 lam~)')
# coherent two-point: Choi root fidelity between N_{p,th} and N_{p,th+D}
for p in [1e-2,1e-3]:
    for D in [1e-3,1e-2]:
        sF=p/2+np.sqrt((1-p/2)**2-(1-p)**2*np.sin(D/2)**2)
        print(f'p={p} Dtheta={D}: 1-sqrtF={1-sF:.3e}, ratio to sin^2(D/2)/2 = {(1-sF)/(np.sin(D/2)**2/2):.4f}; diamond={dnorm(Ndep(p,D)-Ndep(p,0),2,2):.3e} vs 2(1-p)sin(D/2)={2*(1-p)*np.sin(D/2):.3e}')
