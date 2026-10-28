from pathlib import Path
import numpy as np, itertools; from lib import *
exec(open(Path(__file__).resolve().parent / 'budget.py').read().split('def ad(g)')[0].split('from lib import *')[1])
def rx(t): return np.array([[np.cos(t/2),-1j*np.sin(t/2)],[-1j*np.sin(t/2),np.cos(t/2)]])
print("Two independent tomographies of the SAME gate Rx(pi/2)+dep p (true F_hat=1). median 1-F_hat ; frac estimates rank-deficient")
for p in [1e-2,1e-3,1e-4]:
    J=dep(choi([rx(np.pi/2)]),p); row=[]
    for N in [10**3,10**4,10**5,10**6,10**7]:
        v=[];rk=0
        for _ in range(60):
            a=2*tomo(J/2,N); b=2*tomo(J/2,N); v.append(1-geo(a,b)); rk+= (np.linalg.eigvalsh(a)[0]<1e-12)
        row.append(f"N={N:.0e}:{np.median(v):.3f}(rd {rk/60:.0%})")
    print(f"p={p:.0e} "+" ".join(row))
