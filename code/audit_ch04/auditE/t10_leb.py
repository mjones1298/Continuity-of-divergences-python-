import numpy as np
from lib import *
from t5_gen import struct_instance
rng=np.random.default_rng(41)
def shorted(A,B):
    # shorted operator of A to range(B) (Anderson-Trapp): A - A Pp (Pp A Pp)^+ Pp A, Pp = projection onto ker B
    w,V=eigh(herm(B)); Vk=V[:,w<1e-9*max(1,w.max())]
    Pp=Vk@Vk.conj().T
    M=Pp@A@Pp
    return herm(A-A@Pp@np.linalg.pinv(M,hermitian=True,rcond=1e-10)@Pp@A)
mx=0;mx2=0;mx3=0
for trial in range(300):
    dY,dX=[(2,2),(3,2),(2,3)][rng.integers(3)]
    N=dY*dX
    ns=rng.integers(0,N); ne1=rng.integers(0,N-ns+1); ne2=rng.integers(0,N-ns+1)
    if ns+ne1+ne2>N or ns+ne1==0 or ns+ne2==0: continue
    J=struct_instance(dY,dX,rng,ns,ne1,ne2)
    S=Setup(*J,dY,dX);q=sec11(S)
    W=S.W
    Om1ac=shorted(S.Om[0],S.Om[1]); Om2ac=shorted(S.Om[1],S.Om[0])
    mx=max(mx,np.abs(Om1ac-W@q['QR']@S.C[0]@W.conj().T).max())
    mx=max(mx,np.abs(Om2ac-W@q['QR']@S.C[1]@W.conj().T).max())
    # singular parts
    mx2=max(mx2,np.abs((S.Om[0]-Om1ac)-W@q['Q1']@W.conj().T).max())
    mx2=max(mx2,np.abs((S.Om[1]-Om2ac)-W@q['Q2']@W.conj().T).max())
    # h_ac from definition
    G=gmean_frame(S.Om[0],S.Om[1])
    X=Om1ac+Om2ac-2*G
    hac2=opn(ptrace_Y(X,dY,dX))
    mx3=max(mx3,abs(hac2-q['hac2']))
print(mx,mx2,mx3)
