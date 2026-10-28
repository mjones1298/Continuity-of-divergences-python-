# Logarithmic mean: slow (1/ln(1/eta)) convergence of the eta-regularisation to W g(T) W^*; check err*ln(1/eta)->const
import numpy as np, mpmath as mp, sys
sys.argv=[sys.argv[0],'3']
exec(open('test_fundamental.py').read().split("worst = {}")[0])
mp.mp.dps = 500
rng = np.random.default_rng(5)
kp = [mpm(rng.normal(size=(2,2))+1j*rng.normal(size=(2,2)))]
kq = [mpm(rng.normal(size=(2,2))+1j*rng.normal(size=(2,2)))]
Jp, Jq = choi(kp), choi(kq)
W, Wp, T = arn(Jp, Jq)
name, f, f0, finf = [c for c in CONN if c[0]=="logarithmic"][0]
G = W*g_of_T(T,f,f0,finf)*W.H
print("spec T", [mp.nstr(e,4) for e in mp.eigh(herm(T),eigvals_only=True)], " ||G||=",mp.nstr(fro(G),5))
for k in [10,20,40,80,160,300]:
    eta = mp.mpf(10)**(-k)
    M = ka_pd(Jp+eta*mp.eye(4), Jq+eta*mp.eye(4), f)
    e = fro(M-G)
    print(k, mp.nstr(e,6), "err*ln(1/eta)=", mp.nstr(e*mp.log(1/eta),6))
