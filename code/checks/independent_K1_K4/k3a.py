# K3 interpretation (i): Phi = N_p itself, ratio (1-Fhat(N_p,N'))/d, sup over d, for coherent and Pauli-X families
import numpy as np, mpmath as mp
from lib import *
mp.mp.dps=60
def closed(p,d):
    x=4*(1-p)**2*mp.sin(d/2)**2/(p*(4-3*p)); return p/2+(1-p/2)/mp.sqrt(1+x)
for p in ['0.01','0.001']:
    pm=mp.mpf(p)
    J0=mp_choi_Np(p)
    # coherent: d = diamond of rotation diff; compute diamond numerically at a few deltas to calibrate
    best=(0,0)
    for u in np.linspace(0.2,4,60):
        de=2*u*float(mp.sqrt(pm))
        dd=diamond_sdp(J_Np(float(p),de)-J_Np(float(p)),2,2)
        r=(1-closed(pm,de))/dd
        if r>best[0]: best=(r,de,dd)
    print('p',p,'coherent: sup (1-F)/d =',mp.nstr(best[0],6),' *sqrt(p)=',mp.nstr(best[0]*mp.sqrt(pm),6),' *p=',mp.nstr(best[0]*pm,6),' at delta',best[1])
    # Pauli-X transfer: N'=(1-t)N_p + t X o N_p
    Jx=choi_from_map(lambda r: X@Np_map(float(p))(r)@X,2)
    Jn=J_Np(float(p)); JxM=to_mp(Jx)
    best=(0,0)
    for t in np.logspace(-6,-0.3,40):
        Jp=(1-mp.mpf(t))*J0+mp.mpf(t)*JxM
        F=Fhat_mp(J0,Jp); dd=t*diamond_sdp(Jx-Jn,2,2)
        r=(1-F)/dd
        if r>best[0]: best=(r,t)
    print('   X-mix (1-t)N+tXN, Phi=N_p: sup (1-F)/d =',mp.nstr(best[0],6),'*p=',mp.nstr(best[0]*pm,5),'at t',best[1])
    # Pauli weight transfer from I to X and from X to I (Pauli channel picture), Phi=N_p
    a=pm/4; qI=1-3*a
    best=(0,0)
    for s in np.concatenate([-np.logspace(-8,np.log10(float(a)),30), np.logspace(-8,-0.5,30)]):
        s=mp.mpf(s)
        F=mp.sqrt(qI*(qI-s))+mp.sqrt(a*(a+s))+2*a
        r=(1-F)/(2*abs(s))
        if r>best[0]: best=(r,s)
    print('   Pauli I<->X transfer, Phi=N_p (Bhattacharyya): sup ratio',mp.nstr(best[0],6),'at s',mp.nstr(best[1],4))
g=lambda u:(1-1/mp.sqrt(1+u**2))/u
um=mp.findroot(lambda u: mp.diff(g,u),1.0); print('analytic coherent constant max g/2 =',mp.nstr(g(um)/2,8),' 1/(3sqrt3)=',mp.nstr(1/(3*mp.sqrt(3)),8))
