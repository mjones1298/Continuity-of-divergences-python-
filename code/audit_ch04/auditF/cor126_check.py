"""M4: U_n = exp(iH/n) with HV not in CV can still give U_nV in CV for isolated n (Cor 12.6 proof)."""
import numpy as np
from scipy.linalg import expm
X=np.array([[0,1],[1,0]],complex); V=np.array([[1],[0]],complex)   # Y=C^2, X=C, isometric channel Phi = V(.)V^*
H=np.pi*X; print("H V = ", (H@V).ravel(), "(not parallel to V)")
for n in [1,2,3,10]:
    U=expm(1j*H/n); W=U@V; par=abs(abs(np.vdot(V,W))-np.linalg.norm(W)*np.linalg.norm(V))<1e-12
    print("n=%d  U_n V = %s   U_nV in CV: %s"%(n,np.round(W.ravel(),4),par))
