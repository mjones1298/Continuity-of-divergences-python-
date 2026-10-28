import sys; sys.path.insert(0,'.')
from lib import *
from dn import diamond_herm
def abs_h(K):
    w,V=eigh(herm(K)); return (V*np.abs(w))@V.conj().T
def adj1(kr,T):
    # Psi_T^*(1) = sum conj(T_kl) V_k^* V_l
    r=len(kr); dX=kr[0].shape[1]
    M=np.zeros((dX,dX),complex)
    for k in range(r):
        for l in range(r):
            M+=np.conj(T[k,l])*kr[k].conj().T@kr[l]
    return M
worst1=0;worst2=0;sat=0
for trial in range(25):
    dX=rng.integers(1,3); dY=rng.integers(1,3)
    r=rng.integers(1,dX*dY+1)
    kr=rand_kraus(r,dY,dX); W=choi_w(kr,dX)
    T1=effect(r); T2=effect(r)
    K=T1-T2
    lhs=diamond_herm(W@K@W.conj().T,dY,dX)
    mid=opnorm(adj1(kr,abs_h(K)))
    # also diamond of |K| map via SDP (should equal mid)
    mid_sdp=diamond_herm(W@abs_h(K)@W.conj().T,dY,dX)
    Om=opnorm(adj1(kr,np.eye(r)))
    rhs=Om*opnorm(K)
    worst1=max(worst1,lhs/mid); worst2=max(worst2,mid/rhs)
    assert abs(mid-mid_sdp)<1e-4*max(1,mid), (mid,mid_sdp)
    assert lhs<=mid+1e-5 and mid<=rhs+1e-6,(lhs,mid,rhs)
print("ok; max lhs/mid",worst1,"max mid/rhs",worst2)
