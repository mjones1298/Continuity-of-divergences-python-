import numpy as np, warnings
from schur import schur_norm
from lib import Mdd
warnings.filterwarnings("ignore")
rng=np.random.default_rng(3)
g=lambda t:t**1.3; gp=lambda t:1.3*t**0.3
Lam=np.array([0.1,0.35,0.6,1.0])
M=Mdd(g,gp,Lam)
i1=[0,0,1,2,2,2,3]; M2=M[np.ix_(i1,i1)]
print("Schur(M)=",schur_norm(M)," Schur(M with repeated rows/cols)=",schur_norm(M2))
# permutation invariance
p=rng.permutation(4); print("perm:",schur_norm(M[np.ix_(p,p)]))
# basis dependence of Schur norm itself (non-invariance under unitary conjugation of the multiplier matrix): illustrate
Q=np.linalg.qr(rng.normal(size=(4,4)))[0]
print("Schur norm of Q M Q^T (different basis -> different multiplier):", schur_norm(Q@M@Q.T))
