from pathlib import Path
import itertools
import numpy as np; from lib import *
exec(open(Path(__file__).resolve().parent / 'budget.py').read().split('def ad(g)')[0].split('from lib import *')[1])
def rx(t): return np.array([[np.cos(t/2),-1j*np.sin(t/2)],[-1j*np.sin(t/2),np.cos(t/2)]])
def trunc(R,r):
    w,V=np.linalg.eigh(R); w[:-r]=0; w=w/w.sum(); return (V*w)@V.conj().T
def uhl(a,b): return uhlF(a,b)
print("Same gate Rx(pi/2)+dep p, two tomographies, then low-rank (rank r) projection as in compressed-sensing / 'nearest-unitary' fits")
for p in [1e-2,1e-3]:
    J=dep(choi([rx(np.pi/2)]),p)
    for N in [10**4,10**6]:
        for r in [1,2,4]:
            g=[];u=[]
            for _ in range(40):
                a=trunc(tomo(J/2,N),r); b=trunc(tomo(J/2,N),r); g.append(geo(2*a,2*b)); u.append(uhl(a,b))
            print(f"p={p:.0e} N={N:.0e} rank={r}: median F_hat={np.median(g):.4f} (max {np.max(g):.2e}) | Choi Uhlmann F={np.median(u):.5f}")
