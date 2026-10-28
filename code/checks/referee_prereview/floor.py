import numpy as np; from lib import *
rng=np.random.default_rng(11)
def rx(t): return np.array([[np.cos(t/2),-1j*np.sin(t/2)],[-1j*np.sin(t/2),np.cos(t/2)]])
def randchoi():
    G=rng.normal(size=(8,2))+1j*rng.normal(size=(8,2)); Q,_=np.linalg.qr(G); return choi([Q[2*i:2*i+2,:] for i in range(4)])
def dia_ub(J1,J2): # ||J1-J2||_1 >= diamond/... use Choi trace norm (unnormalized) as proxy; diamond <= d*||J/d diff||_1
    return np.sum(np.abs(np.linalg.eigvalsh(J1-J2)))
print("Question: F_hat(Phi,Psi) for Phi=Psi=Rx(pi/2)+dep p (F=1). Perturb Psi by eps*(random channel - Psi). Median 1-F_hat over 200 draws.")
print("p, then eps* where median(1-F)>=0.1, and eps*/p, eps*/sqrt(p)")
for p in [1e-2,1e-3,1e-4,1e-5]:
    J=dep(choi([rx(np.pi/2)]),p)
    eps_grid=np.logspace(-7,-0.5,40); es=None
    for e in eps_grid:
        v=[1-geo(J,(1-e)*J+e*randchoi()) for _ in range(150)]
        if np.median(v)>=0.1: es=e;break
    # coherent: over-rotation delta
    ds=None
    for dl in np.logspace(-5,0,200):
        if 1-geo(J,dep(choi([rx(np.pi/2+dl)]),p))>=0.1: ds=dl;break
    print(f"p={p:.0e}: random eps*={es:.2e} eps*/p={es/p:.2f} | coherent delta*={ds:.2e} delta*/sqrt(p)={ds/np.sqrt(p):.2f}")
