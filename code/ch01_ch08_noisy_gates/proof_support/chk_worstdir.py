import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))  # shared modules in code/common (glib.py)
import numpy as np, glib as g, scipy.linalg as sla
exec(open(Path(__file__).resolve().parent / 'chk_localmod.py').read().split("rng=np.random.default_rng(0)")[0])
I=np.eye(2); Yp=np.array([[0,-1j],[1j,0]]); Z=np.diag([1.,-1]).astype(complex)
P=[I,X,Yp,Z]; basis=np.array([g.vec(s)/np.sqrt(2) for s in P]).T   # Bell basis (Choi of Paulis)
p=1e-3; delta=2*np.sqrt(p); J1=gate(p,delta); J2=gate(p,0.0)
xs=[np.array([np.cos(a/2),np.exp(1j*b)*np.sin(a/2)]) for a in np.linspace(0,np.pi,9) for b in [0,np.pi/2,np.pi,3*np.pi/2]]
best=(0,None,None)
for x in xs:
    v,E=worst_rate(grad(J1,J2,x))
    if v>best[0]: best=(v,E,x)
v,E,x=best
print("rate*p",v*p,"x",np.round(x,3))
Eb=basis.conj().T@E@basis
np.set_printoptions(precision=3,suppress=True,linewidth=150)
print("E in Bell/Pauli basis (I,X,Y,Z):\n",Eb)
print("eig E",np.linalg.eigvalsh(E))
# minimizer x of F: eigvec of TrY(J1#J2)
M=g.trY(g.gmean(J1,J2),2,2); print("TrY(J1#J2) eig",np.linalg.eigvalsh(M))
