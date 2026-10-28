import numpy as np, mpmath as mp
from qlib import *
rng=np.random.default_rng(5)
P=lambda v: np.outer(v,np.conj(v))
for th in [1e-3,0.3]:
    J0=np.kron(P(np.array([1,0])),I2); Jt=np.kron(P(np.array([np.cos(th),np.sin(th)])),I2)
    print('C1 th',th, fid_sub(J0,Jt,2,2), fid_sub((J0+Jt)/2,(J0+Jt)/2,2,2))
def JAD(g): v0=np.array([1,0,0,np.sqrt(1-g)]); v1=np.array([0,np.sqrt(g),0,0]); return P(v0)+P(v1)
print('AD',fid_sub(JAD(.5),JAD(.499),2,2),fid_sub(JAD(.9),JAD(.9),2,2))
# random: low-rank pairs, compare to SDP and to mp built from exact-rank construction
mx=0
for k in range(15):
    r1,r2=rng.choice([1,2,3],2)
    J1=choi_from_kraus(rand_channel_kraus(2,2,r1,rng)); J2=choi_from_kraus(rand_channel_kraus(2,2,r2,rng))
    # make them share support sometimes: mix
    if k%2: J2=0.5*J2+0.5*J1
    a=fid_sub(J1,J2,2,2); b=fid_sdp(J1,J2,2,2)
    mx=max(mx,abs(a-b)); print(r1,r2,k%2,f"sub={a:.8f} sdp={b:.8f}")
print('max |sub-sdp|',mx)
# full-rank check vs plain
J1=choi_from_kraus(rand_channel_kraus(2,2,4,rng)); J2=choi_from_kraus(rand_channel_kraus(2,2,4,rng))
print('fullrank', fid_sub(J1,J2,2,2), fid_np(J1,J2,2,2), float(fid_mp(J1,J2,2,2,'0',60)))
