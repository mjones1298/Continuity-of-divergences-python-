import numpy as np, sys
sys.path.insert(0,'.')
from lib import *
src=open('appendix_final.py').read().split("ps=[1,1.5")[0]
exec(src)
rng=np.random.default_rng(5)
worst=0; worst_by={}
for trial in range(60):
    d=int(rng.integers(2,4)); m=int(rng.integers(1,4))
    ra=int(rng.integers(1,d+1)); rb=int(rng.integers(1,d+1)); rc=int(rng.integers(1,m+1))
    def psd(n,r):
        B=rng.normal(size=(n,r))+1j*rng.normal(size=(n,r)); return B@B.conj().T
    A,B,C=psd(d,ra),psd(d,rb),psd(m,rc)
    for name in ['geo','harm','mix','alpha0.3','alpha0.8','skew','log']:
        cn=conns[name]
        L,_=mean_choi(np.kron(A,C),np.kron(B,C),cn)
        R=np.kron(mean_choi(A,B,cn)[0],C)
        e=np.linalg.norm(L-R)/max(1,np.linalg.norm(R)); worst_by[name]=max(worst_by.get(name,0),e); worst=max(worst,e)
print("snapped-spectrum mean: max rel error", worst, worst_by)
