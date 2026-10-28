import numpy as np
from lib import *
rng=np.random.default_rng(9)
def chk(J,dY,dX,label):
    S=Setup(*J,dY,dX);q=sec11(S,tol=1e-8)
    F1=Fhat(J[0],J[1],dY,dX);F2=Fhat(J[2],J[3],dY,dX)
    R=R_gsharp(S,q)
    ok=abs(F1-F2)<=R+1e-8
    print(label,'F1=%.4f F2=%.4f R=%.4f'%(F1,F2,R),'OK' if ok else 'VIOLATION', 'Q dims',[int(round(np.trace(q[k]).real)) for k in ('Q1','QR','Q2')])
for (dY,dX) in [(2,2),(1,3),(3,1),(2,3)]:
    N=dY*dX
    Z=np.zeros((N,N),complex)
    P=lambda k: rand_psd(N,k,rng)
    a,b,c,d=P(1),P(2),P(2),P(1)
    chk([Z,Z,c,d],dY,dX,f'{dY}x{dX} first pair zero')
    chk([a,b,Z,Z],dY,dX,f'{dY}x{dX} second pair zero')
    chk([a,Z,c,d],dY,dX,f'{dY}x{dX} Psi1=0')
    chk([a,b,a,b],dY,dX,f'{dY}x{dX} identical')
    chk([a,b,b,a],dY,dX,f'{dY}x{dX} swapped')
    chk([Z,b,Z,d],dY,dX,f'{dY}x{dX} Phi_i=0')
    chk([a,a,c,c],dY,dX,f'{dY}x{dX} Phi=Psi both')
