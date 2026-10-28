# (W transfer): (W A W*) s (W B W*) = W (A s B) W*  for singular PSD A,B and generic injective W (non-isometric), mp precision
import numpy as np, mpmath as mp, sys
sys.argv=[sys.argv[0],'9']
exec(open('test_fundamental.py').read().split("worst = {}")[0])
mp.mp.dps = 120
rng=np.random.default_rng(12)
r,n=3,6
def rnd(a,b): return mpm(rng.normal(size=(a,b))+1j*rng.normal(size=(a,b)))
W=rnd(n,r)
def psd(rank):
    X=rnd(r,rank); return X*X.H
cases={"A rank1,B rank1 (indep)":(psd(1),psd(1)),"A rank2,B rank2":(psd(2),psd(2)),"A rank1 = B rank1 (same)":None,"A rank 3, B rank 1":(psd(3),psd(1))}
X1=rnd(r,1); cases["A rank1 = B rank1 (same)"]=(X1*X1.H, 2*X1*X1.H)
eta=mp.mpf(10)**(-50)
for name,(A,B) in cases.items():
    for (cn,f,_,_) in [CONN[0],CONN[3],CONN[4],CONN[5],CONN[1]]:
        lhs=ka_pd(W*A*W.H+eta*mp.eye(n), W*B*W.H+eta*mp.eye(n), f)
        rhs=W*ka_pd(A+eta*mp.eye(r),B+eta*mp.eye(r),f)*W.H
        print(f"{name:28s} {cn:14s} ||lhs-rhs||_F={mp.nstr(fro(lhs-rhs),3)}   ||rhs||={mp.nstr(fro(rhs),3)}")
