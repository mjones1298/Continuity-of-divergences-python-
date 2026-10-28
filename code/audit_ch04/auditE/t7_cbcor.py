import numpy as np
from lib import *
from diamond import diamond
rng=np.random.default_rng(5)
def same_supp(dY,dX,m,rng):
    N=dY*dX
    U,_=np.linalg.qr(rng.standard_normal((N,N))+1j*rng.standard_normal((N,N)))
    return U[:,:m]
worst1=0;worst2=0;worst3=0
for trial in range(60):
    dY,dX=[(2,2),(2,3),(3,2)][rng.integers(3)]
    N=dY*dX; m=rng.integers(2,N+1)
    U=same_supp(dY,dX,m,rng)
    # base pair, perturb second pair slightly / or not slightly
    eps=rng.choice([0.01,0.1,0.5,1.0])
    G=[rand_psd(m,m,rng) for _ in range(4)]
    G2=[G[0]+eps*rand_psd(m,m,rng,0.5), G[1]+eps*rand_psd(m,m,rng,0.5)]
    Js=[U@G[0]@U.conj().T,U@G[1]@U.conj().T,U@G2[0]@U.conj().T,U@G2[1]@U.conj().T]
    S=Setup(*Js,dY,dX)
    nu=nu_of_frame(S.W,dY,dX,rng,starts=8)
    d1=diamond(Js[0]-Js[2],dY,dX); d2=diamond(Js[1]-Js[3],dY,dX)
    cbmax=max(d1,d2)
    r1=S.delta*nu**2/cbmax
    worst1=max(worst1,r1)
    # nu(Omega1) <= nu(Omega')
    Wo1=frame(S.Om[0]); nu1=nu_of_frame(Wo1,dY,dX,rng,starts=8)
    worst2=max(worst2,nu1/nu)
    # final
    F1=Fhat(Js[0],Js[1],dY,dX);F2=Fhat(Js[2],Js[3],dY,dX)
    rhs=11*S.nOmb/(4*nu1)*np.sqrt(cbmax)
    worst3=max(worst3,abs(F1-F2)/rhs)
print('delta*nu^2/cb max',worst1,' nu1/nu(Omega\') max',worst2,' F diff / rhs',worst3)
