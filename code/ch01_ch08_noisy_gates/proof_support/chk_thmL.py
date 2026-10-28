import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))  # shared modules in code/common (glib.py)
# Theorem L check: |F(Phi1,Psi)-F(Phi2,Psi)| <= dX*||Phi1-Phi2||_dia/(2*lam), lam=min lambda_min(J(Phi_i)), channels
import numpy as np, glib as g
from scale_law import zeta_lower
rng=np.random.default_rng(3); worst=0
for t in range(60):
    dX,dY=2,2
    p=10**rng.uniform(-3,-1)
    J0=lambda K:(1-p)*g.choi(K)+p*np.eye(4)/2
    K1=g.rand_kraus(rng,dY,dX,rng.integers(1,3)); Kpsi=g.rand_kraus(rng,dY,dX,rng.integers(1,3))
    e=10**rng.uniform(-4,-1)*p; K2=g.rand_kraus(rng,dY,dX,2)
    JA=J0(K1); JB=(1-e)*JA+e*J0(K2); JP=J0(Kpsi)
    lam=min(np.linalg.eigvalsh(JA).min(),np.linalg.eigvalsh(JB).min())
    d=g.dnorm(JA-JB,2,2); dF=abs(g.Fhat(JA,JP,2,2)-g.Fhat(JB,JP,2,2))
    worst=max(worst,dF/(dX*d/(2*lam)))
print("Theorem L: max |dF|/bound =",worst)
m=1e9
for p in np.geomspace(1e-6,1e-2,9):
    for r in np.geomspace(1e-3,1,7):
        d=r*p; z,k=zeta_lower(p,d); m=min(m,z*p/d**2)
print("min zeta_low*p/delta^2 over p in[1e-6,1e-2], delta<=p:",m)
