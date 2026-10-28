import numpy as np, sys
sys.path.insert(0,'.')
from lib import *
rng = np.random.default_rng(13)
conns = make_conns()
n=4
def rpsd(rank): G=rng.normal(size=(n,rank))+1j*rng.normal(size=(n,rank)); return G@G.conj().T
A = rpsd(1); B = rpsd(2)   # generic: ranges intersect trivially
for name in ['geo','harm','mix','skew','alpha0.8','log']:
    cn = conns[name]
    row=[]
    for eps in [1e-2,1e-4,1e-6]:
        worst=0
        for trial in range(100):
            G1=rpsd(n); G2=rpsd(n)
            An=A+eps*G1/np.linalg.norm(G1,2); Bn=B+eps*G2/np.linalg.norm(G2,2)
            L=ka_frame_conn(An,Bn,cn,tol=1e-14)
            worst=max(worst,np.linalg.norm(L-(cn.a*A+cn.b*B),2))
        row.append("eps=%.0e: %.2e"%(eps,worst))
    print("%-9s"%name,row)
