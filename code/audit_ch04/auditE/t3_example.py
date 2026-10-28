import numpy as np
from lib import *
Z=np.diag([1.,-1.]);X=np.array([[0,1.],[1,0]])
U0=(Z+X)/np.sqrt(2);U1=(Z-X)/np.sqrt(2)
for d in [0.4,0.2,0.1,0.05,0.01,0.001]:
    T0=np.diag([d,1-d])
    A=[0.5*(T0+s*d*U0) for s in (1,-1)]
    B=[0.5*(np.eye(2)-T0+s*d*U1) for s in (1,-1)]
    mins=[np.linalg.eigvalsh(M)[0] for M in A+B]
    dets=[np.linalg.det(M) for M in A+B]
    A1,A2=A;B1,B2=B
    C1=A1+B1;C2=A2+B2
    assert opn(C1+C2-np.eye(2))<1e-12
    gam=opn(C1-0.5*np.eye(2));dl=max(opn(A1-A2),opn(B1-B2))
    c1h=fun_herm(C1,lambda w:1/np.sqrt(w)); c2h=fun_herm(C2,lambda w:1/np.sqrt(w))
    T1=herm(c1h@A1@c1h);T2=herm(c2h@A2@c2h)
    tau=opn(T1-T2)
    print(f"d={d} mineig={min(mins):.3e} det(A2)={dets[1]:.4e} pred={(2-np.sqrt(2))/8*d*(1-2*d):.4e}  tau/delta={tau/dl:.6f} 2sqrt2={2*np.sqrt(2):.6f} gamma/delta={gam/dl:.4f} bound={2*np.sqrt(2)/np.sqrt(1-4*gam**2):.5f} comm bound 2/(1-2d)={2/(1-2*d):.5f}")
