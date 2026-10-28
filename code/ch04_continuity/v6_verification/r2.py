# R2: Theorem 4.6 asymptotically attained: Phi2=(1-s)Phi1+s Psi1, Psi2=s Phi1+(1-s)Psi1, (Phi1,Psi1) mutually singular channels.
import random, mlib as m, mpmath as mp
mp.mp.dps = 40
rng = random.Random(8)
for dX, dY in [(1,2),(2,2),(2,3)]:
    # mutually singular channels: random isometric channels (rank-one Choi, generic ranges meet trivially)
    J1 = m.choi_kraus(m.rand_channel_kraus(dX, dY, 1, rng), dX, dY)
    K1 = m.choi_kraus(m.rand_channel_kraus(dX, dY, 1, rng), dX, dY)
    F1 = m.Fhat(J1, K1, dY, dX)
    for s in ['0.1', '0.01', '0.0001']:
        s = mp.mpf(s)
        J2 = (1-s)*J1 + s*K1; K2 = s*J1 + (1-s)*K1
        F2 = m.Fhat(J2, K2, dY, dX)
        # ARN derivatives w.r.t. Omega = J1+K1
        W = m.frame(J1+K1); Wp = m.pinv_frame(W)
        T1 = Wp*J1*m.dag(Wp); T2 = Wp*J2*m.dag(Wp)
        bound = 2*mp.sqrt(m.opnorm(T1-T2))       # ||Omega^*(1)|| = 2 for channels
        print(f'dX={dX} dY={dY} s={mp.nstr(s,2)}: F1={mp.nstr(F1,3)}, F2={mp.nstr(F2,12)}, sqrt(s(1-s))*2={mp.nstr(2*mp.sqrt(s*(1-s)),12)}, |dF|/bound={mp.nstr(abs(F2-F1)/bound,12)}, sqrt(1-s)={mp.nstr(mp.sqrt(1-s),12)}')
