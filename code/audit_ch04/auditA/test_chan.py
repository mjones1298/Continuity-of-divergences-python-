import numpy as np
from core import *
rng=np.random.default_rng(4)
def crand(*s): return rng.normal(size=s)+1j*rng.normal(size=s)
def rand_channel(n,dy,dx):
    K=[crand(dy,dx) for _ in range(n)]
    S=sum(k.conj().T@k for k in K); w,V=np.linalg.eigh(S); Sih=(V/np.sqrt(w))@V.conj().T
    return [k@Sih for k in K]
for dx,dy in [(2,2),(3,2),(2,3)]:
    Kp,Kq=rand_channel(3,dy,dx),rand_channel(2,dy,dx)
    G,T,_=gmean_frame(choi(Kp,dx),choi(Kq,dx))
    M=ptrace_out(G,dy,dx); w=np.linalg.eigvalsh((M+M.conj().T)/2)
    print(dx,dy,"eig Tr_Y(J(Phi#Psi)) =",np.round(w,4),"  (<=1: trace-nonincreasing; !=1: not TP)  Fhat in [0,1]:",0<=w[0]<=1)
