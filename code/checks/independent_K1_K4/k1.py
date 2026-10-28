import numpy as np, mpmath as mp
from lib import *
mp.mp.dps=50
rng=np.random.default_rng(12345)
# (a) ||J(Psi')-J(Psi)||_inf <= dX * ||Psi'-Psi||_diamond
worst_a=0; worst_ratio=0
for trial in range(40):
    r=rng.integers(1,5); r2=rng.integers(1,5)
    K1=random_channel_kraus(2,2,r,rng); K2=random_channel_kraus(2,2,r2,rng)
    J1=choi_from_kraus(K1,2); J2=choi_from_kraus(K2,2)
    # also near pairs
    if trial%2==0:
        t=10**rng.uniform(-4,-1); J2=(1-t)*J1+t*J2
    Dl=J2-J1
    dn=diamond_sdp(Dl,2,2)
    nin=np.linalg.norm(Dl,2); ntr=np.abs(np.linalg.eigvalsh(Dl)).sum()
    worst_a=max(worst_a,nin/dn); worst_ratio=max(worst_ratio,ntr/dn)
print('(a) max ||DJ||_inf/diamond over 40 random pairs =',worst_a,' (claim <= 2)')
print('    max ||DJ||_1/diamond =',worst_ratio,' (known <= dX=2)')
# Extremal: Psi'=Psi∘X-type? unitary pairs: J diff for U vs V
for th in [1e-3,0.1,1.0,np.pi/2]:
    J1=choi_from_kraus([I2],2); J2=choi_from_kraus([Rz(th)],2)
    print('  Rz(%g) vs I: ||DJ||_inf/diamond='%th, np.linalg.norm(J2-J1,2)/diamond_sdp(J2-J1,2,2))

# (b) Theorem L on random full-rank qubit pairs (mpmath for Fhat)
def check_L(J_Phi,J_Psi,J_Psip,d):
    lam=min(mp_eigvals_h(J)[0] for J in (J_Phi,J_Psi,J_Psip))
    F=Fhat_mp(J_Phi,J_Psi); Fp=Fhat_mp(J_Phi,J_Psip)
    bound=(1+2*mp.mpf(d)/lam)*F
    return Fp, F, bound, lam
viol=0; minslack=mp.inf; cnt=0
for trial in range(60):
    Ks=[random_channel_kraus(2,2,4,rng) for _ in range(3)]
    Js=[choi_from_kraus(K,2) for K in Ks]
    # mix with depolarizing to control lam
    eps=10**rng.uniform(-3,-0.5)
    Jdep=np.eye(4)/2
    JPhi=(1-eps)*Js[0]+eps*Jdep; JPsi=(1-eps)*Js[1]+eps*Jdep
    t=10**rng.uniform(-4,-0.5)
    JPsip=(1-t)*JPsi+t*((1-eps)*Js[2]+eps*Jdep)
    d=diamond_sdp(JPsip-JPsi,2,2)
    for (A,B,Bp) in [(JPhi,JPsi,JPsip),(JPhi,JPsip,JPsi)]:
        Fp,F,bd,lam=check_L(to_mp(A),to_mp(B),to_mp(Bp),d)
        cnt+=1
        slack=(bd-Fp)/F
        minslack=min(minslack,slack)
        if Fp>bd*(1+mp.mpf('1e-7')): viol+=1; print('VIOLATION',Fp,bd)
print('(b) random full-rank: checks',cnt,'violations',viol,'min relative slack (bound-F\')/F =',mp.nstr(minslack,5))

# N_p families: Phi=N_{p,delta1}, Psi=N_p, Psi'=N_{p,delta2} or Pauli-mixed; also depolarized-gate consequence |dF|<=4d/p
for p in [1e-1,1e-2,1e-3]:
    maxr=0; maxr_tight=0
    for trial in range(25):
        d1=rng.uniform(0,1)*np.sqrt(p)*rng.choice([0.1,1,3])
        JPhi=mp_choi_Np(p,d1); JPsi=mp_choi_Np(p)
        kind=trial%3
        if kind==0:
            d2=10**rng.uniform(-5,-1)
            JPsip_np=J_Np(p,d2); JPsip=mp_choi_Np(p,d2)
        elif kind==1:
            t=10**rng.uniform(-5,-1)*p
            Jx=choi_from_map(lambda r: X@Np_map(p)(r)@X,2)
            JPsip_np=(1-t)*J_Np(p)+t*Jx; JPsip=to_mp(JPsip_np)
        else:
            K=random_channel_kraus(2,2,4,rng); t=10**rng.uniform(-5,-1)*p
            JPsip_np=(1-t)*J_Np(p)+t*choi_from_kraus(K,2); JPsip=to_mp(JPsip_np)
        d=diamond_sdp(JPsip_np-J_Np(p),2,2)
        Fp,F,bd,lam=check_L(JPhi,JPsi,JPsip,d)
        Fp2,F2,bd2,lam2=check_L(JPhi,JPsip,JPsi,d)
        if Fp>bd*(1+mp.mpf('1e-7')) or Fp2>bd2*(1+mp.mpf('1e-7')): print('VIOLATION N_p',p,kind)
        r=abs(Fp-F)/d
        maxr=max(maxr,r*p)
    print(f'  p={p}: max |dFhat|/d * p over family = {mp.nstr(maxr,6)} (consequence claims <= 4)')
