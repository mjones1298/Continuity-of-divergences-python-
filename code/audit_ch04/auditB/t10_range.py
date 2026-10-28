import sys; sys.path.insert(0,'.')
from lib import *
from core import gmean_frame
import scipy.linalg as sl
def rank(M,tol=1e-6): 
    s=np.linalg.svd(M,compute_uv=False); return int((s>tol*max(1,s[0])).sum())
for trial in range(6):
    n=6
    # common vector(s)
    c=rng.integers(0,3)
    common=rand_c(n,c)
    ra=rng.integers(0,3); rb=rng.integers(0,3)
    U=np.hstack([common,rand_c(n,ra)]); V=np.hstack([common,rand_c(n,rb)])
    A=U@np.diag(rng.uniform(0.5,2,U.shape[1]))@U.conj().T if U.shape[1] else np.zeros((n,n))
    B=V@np.diag(rng.uniform(0.5,2,V.shape[1]))@V.conj().T if V.shape[1] else np.zeros((n,n))
    if U.shape[1]+V.shape[1]==0: continue
    G,T,Sh=gmean_frame(A,B)
    # dim of ran A ∩ ran B = rank A + rank B - rank(A+B)
    inter=rank(A)+rank(B)-rank(A+B)
    print("rankA",rank(A),"rankB",rank(B),"dim intersection",inter,"rank(A#B)",rank(G))
