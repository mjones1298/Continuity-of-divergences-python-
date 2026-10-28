import sys; sys.path.insert(0,'.')
from lib import *
def xlogx(x): return np.where(x>0,x*np.log(np.where(x>0,x,1)),0.)
fJS=lambda x: xlogx(x)-(1+x)*np.log((1+x)/2)
fH=lambda x:(1-np.sqrt(x))**2
fLC=lambda x:(x-1)**2/(x+1)
fHa=lambda x:(1-x**0.3)/0.7
def minconv(f,trials=4000,n=3):
    worst=1e9
    for _ in range(trials):
        A=rand_c(n,n);A=A@A.conj().T*rng.uniform(0.1,5)
        B=rand_c(n,n);B=B@B.conj().T*rng.uniform(0.1,5)
        lhs=psd_fun2(herm((A+B)/2),f)
        rhs=(psd_fun2(A,f)+psd_fun2(B,f))/2
        worst=min(worst,np.linalg.eigvalsh(herm(rhs-lhs))[0])
    return worst
for nm,f in [('JS',fJS),('H',fH),('LC',fLC),('Ha',fHa)]:
    print(nm,[minconv(f,n=n) for n in (2,3)])
