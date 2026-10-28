import sys; sys.path.insert(0,'.')
from lib import *
from dn import diamond_herm
import warnings; warnings.filterwarnings("ignore")
mn=1e9; mx=0
for trial in range(40):
    dX=rng.integers(1,3); dY=rng.integers(1,3)
    r=rng.integers(1,dX*dY+1)
    kr=rand_kraus(r,dY,dX); W=choi_w(kr,dX)
    G=W.conj().T@W
    lam=np.linalg.eigvalsh(G)[0]
    Om1=opnorm(sum(K.conj().T@K for K in kr))
    K=herm(rand_c(r,r)); 
    if trial%3==0:
        K=effect(r)-effect(r)
    lhs=lam/dX*np.linalg.norm(np.linalg.svd(K,compute_uv=False),1)
    dnorm=diamond_herm(W@K@W.conj().T,dY,dX)
    rhs=Om1*opnorm(K)
    assert lhs<=dnorm*(1+1e-5)+1e-7 and dnorm<=rhs*(1+1e-5)+1e-7,(lhs,dnorm,rhs)
    mn=min(mn,lhs/dnorm); mx=max(mx,dnorm/rhs)
print("ok",mn,mx)
