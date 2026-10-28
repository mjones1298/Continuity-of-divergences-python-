import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))  # shared modules in code/common (glib.py)
# Theorem P3 check: mutually singular channels, F(Phi',Psi') <= 2 cot^2(theta/2) sqrt(d), d = max diamond distance
import numpy as np, glib as g
from scipy.linalg import orth, subspace_angles
rng=np.random.default_rng(12); worst=0; cnt=0
for t in range(120):
    dX,dY=2,2
    K1=g.rand_kraus(rng,dY,dX,rng.integers(1,3)); K2=g.rand_kraus(rng,dY,dX,rng.integers(1,3))
    J1,J2=g.choi(K1),g.choi(K2)
    if np.linalg.matrix_rank(J1+J2,tol=1e-9)<np.linalg.matrix_rank(J1,tol=1e-9)+np.linalg.matrix_rank(J2,tol=1e-9): continue
    th=subspace_angles(orth(J1,rcond=1e-10),orth(J2,rcond=1e-10)).min()
    e=10**rng.uniform(-6,-1)
    J1p=(1-e)*J1+e*g.choi(g.rand_kraus(rng,dY,dX,4)); J2p=(1-e)*J2+e*g.choi(g.rand_kraus(rng,dY,dX,4))
    d=max(g.dnorm(J1p-J1,dY,dX),g.dnorm(J2p-J2,dY,dX))
    F=g.Fhat(J1p,J2p,dY,dX); B=2*np.sqrt(d)/np.tan(th/2)**2
    worst=max(worst,F/B); cnt+=1
print(cnt,"mutually singular instances; max F/bound =",worst)
