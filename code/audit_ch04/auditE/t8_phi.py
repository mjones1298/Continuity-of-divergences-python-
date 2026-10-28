import numpy as np
from lib import *
from t5_gen import struct_instance
rng=np.random.default_rng(21)

# family of phi with (func, modulus bound omega(t), sup norm on [0,1])
alpha=0.3
fam={
 't':(lambda t:t, lambda d:d, 1.0),
 'gsharp':(lambda t:np.sqrt(np.clip(t*(1-t),0,None)), lambda d:np.sqrt(d), 0.5),
 'halpha':(lambda t:np.clip(t,0,1)**alpha*np.clip(1-t,0,1)**(1-alpha), lambda d:d**alpha+d**(1-alpha), 1.0),
 'const1':(lambda t:np.ones_like(t), lambda d:0.0*d, 1.0),
 'tq':(lambda t:t*(1-t), lambda d:d, 0.25),
 'harm':(lambda t:t*(1-t), lambda d:d, 0.25),   # parallel sum kernel g(t)=t(1-t)/... (Kubo-Ando harmonic: g=2t(1-t))
 'arith':(lambda t:np.full_like(t,0.5)+0*t, lambda d:0*d, 0.5),
}
def phi_apply(T,P,f):
    # functional calculus on range(P) extended by 0; T acts on range P
    w,V=eigh(herm(T))
    # restrict to range P
    wP,VP=eigh(herm(P)); U=VP[:,wP>0.5]
    Tm=herm(U.conj().T@T@U)
    return U@fun_herm(Tm,f)@U.conj().T

viol=0;worst={}
for trial in range(1500):
    dY,dX=[(2,2),(3,2),(2,3),(1,3),(3,1)][rng.integers(5)]
    N=dY*dX
    ns=rng.integers(0,N); ne1=rng.integers(0,N-ns+1); ne2=rng.integers(0,N-ns+1)
    if ns+ne1+ne2>N or ns+ne1==0 or ns+ne2==0: continue
    J=struct_instance(dY,dX,rng,ns,ne1,ne2)
    S=Setup(*J,dY,dX);q=sec11(S)
    Q1,QR,Q2=q['Q1'],q['QR'],q['Q2']
    P1=Q1+QR;P2=QR+Q2
    nO=S.nOmb
    for name,(f,om,nf) in fam.items():
        N1=q['a1']@phi_apply(q['T1'],P1,f)@q['a1']
        N2=q['a2']@phi_apply(q['T2'],P2,f)@q['a2']
        lhs=opn(S.Delta(N1)-S.Delta(N2))
        R=nO*om(q['tau_ac'])+nf*(q['ms']+min(2*np.sqrt(2)*nO*q['tanh_ac'],2*np.sqrt(nO)*q['hac']))+S.nOm[0]*om(q['kap1'])+S.nOm[1]*om(q['kap2'])
        if R>1e-12:
            worst[name]=max(worst.get(name,0),lhs/R)
        if lhs>R+1e-9: viol+=1;print('VIOL',name,lhs,R)
print(worst,viol)
