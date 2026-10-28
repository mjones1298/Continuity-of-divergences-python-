import numpy as np
from qlib import *
rng=np.random.default_rng(7)
def Eg_bruteforce(JA,JB,g,n=4000):
    best=0
    for _ in range(n):
        v=rng.normal(size=2)+1j*rng.normal(size=2); v/=np.linalg.norm(v)
        rho=np.outer(v,v.conj()); 
        # pure psi on X(x)R with reduced rho: S = 1 (x) sqrt(rho)^T, rho pure -> sqrt=rho
        w,V=np.linalg.eigh(rho); s=V@np.diag(np.sqrt(np.clip(w,0,None)))@V.conj().T
        S=np.kron(I2,s.T); D=S@(JA-g*JB)@S
        best=max(best,np.clip(np.linalg.eigvalsh(D),0,None).sum())
    return best
# sanity check SDP vs brute force over random rho (not only pure rho)
def Eg_rand_rho(JA,JB,g,n=3000):
    best=0
    for _ in range(n):
        G=rng.normal(size=(2,2))+1j*rng.normal(size=(2,2)); rho=G@G.conj().T; rho/=np.trace(rho).real
        w,V=np.linalg.eigh(rho); s=V@np.diag(np.sqrt(w))@V.conj().T
        S=np.kron(I2,s.T); D=S@(JA-g*JB)@S
        best=max(best,np.clip(np.linalg.eigvalsh(D),0,None).sum())
    return best
J1=choi_from_kraus(rand_channel_kraus(2,2,2,rng)); J2=choi_from_kraus(rand_channel_kraus(2,2,1,rng))
print('SDP vs sampled:',hockey_sdp(J1,J2,1.5,2,2),Eg_rand_rho(J1,J2,1.5))
# gamma=1 consistency: E_1 = ||Phi-Psi||_diamond/2
print('E_1 vs diamond/2:',hockey_sdp(J1,J2,1.0,2,2),diamond(J1-J2,2,2)/2)
worst=0; rows=[]
for k in range(150):
    r=rng.choice([1,2,3,4],4)
    A1,B1=[choi_from_kraus(rand_channel_kraus(2,2,ri,rng)) for ri in r[:2]]
    if k%3==0:  # small perturbations
        e=10**rng.uniform(-3,-0.5)
        A2=(1-e)*A1+e*choi_from_kraus(rand_channel_kraus(2,2,r[2],rng)); B2=(1-e)*B1+e*choi_from_kraus(rand_channel_kraus(2,2,r[3],rng))
    else:
        A2,B2=[choi_from_kraus(rand_channel_kraus(2,2,ri,rng)) for ri in r[2:]]
    g=[0,0.5,1,2,5,20][k%6]
    dE=abs(hockey_sdp(A1,B1,g,2,2)-hockey_sdp(A2,B2,g,2,2))
    rhs=(diamond(A1-A2,2,2)+g*diamond(B1-B2,2,2))/2
    worst=max(worst,dE/rhs); rows.append((dE,rhs,g))
print('C7 random: n=',len(rows),' max ratio dE/rhs =',worst)
# adversarial-ish: Psi fixed, Phi varying in near-extremal direction
# Non-TP counterexample (shows trace-preservation is essential): Phi2 = (1-e)Phi1 (trace-decreasing)
e=0.2; A1=choi_from_kraus(rand_channel_kraus(2,2,1,rng)); B1=choi_from_kraus(rand_channel_kraus(2,2,1,rng)); g=0.
dE=abs(hockey_sdp(A1,B1,g,2,2)-hockey_sdp((1-e)*A1,B1,g,2,2)); rhs=(diamond(A1-(1-e)*A1,2,2))/2
print('non-TP (Phi2=(1-e)Phi1, gamma=0): dE=',dE,' rhs=',rhs,' ratio',dE/rhs)
