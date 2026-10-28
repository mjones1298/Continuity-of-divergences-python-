import numpy as np, scipy.linalg as sla
from scipy.optimize import minimize
rng=np.random.default_rng(0)
z=0.01
k  = lambda t: t + 2*z*z/(t+z)
kp = lambda t: 1 - 2*z*z/(t+z)**2
def fun(A):  # matrix function for Hermitian A
    w,V=np.linalg.eigh(A); return (V*k(w))@V.conj().T
def ratio(X,Y):
    return np.linalg.norm(fun(X)-fun(Y),2)/np.linalg.norm(X-Y,2)
# extremal: X=Z+uH, Y=Z with Z=diag(0,1,0,1), H=[[0,N],[N*,0]] with N=diag-ish; maximize derivative
lam=np.array([0,1,0,1.0])
M=np.array([[ (k(x)-k(y))/(x-y) if abs(x-y)>1e-12 else kp(x) for y in lam] for x in lam])
def negobj(v):
    N=(v[:4]+1j*v[4:8]).reshape(2,2)
    N=N/np.linalg.norm(N,2)
    H=np.zeros((4,4),complex); H[:2,2:]=N; H[2:,:2]=N.conj().T
    return -np.linalg.norm(M*H,2)
best=0
for t in range(30):
    r=minimize(negobj,rng.normal(size=8),method='Nelder-Mead',options={'xatol':1e-10,'fatol':1e-12,'maxiter':20000})
    best=max(best,-r.fun)
print("max ||M o H|| over Hermitian H dilation:",best, "(Schur endpoint norm 1.4002...)")
# direct operator ratio with Z+uH
v=minimize(negobj,rng.normal(size=8),method='Nelder-Mead',options={'maxiter':20000}).x
N=(v[:4]+1j*v[4:8]).reshape(2,2); N/=np.linalg.norm(N,2)
H=np.zeros((4,4),complex); H[:2,2:]=N; H[2:,:2]=N.conj().T
for u in [1e-3,1e-5]:
    Z=np.diag(lam)+0j
    # shift spectrum into [0,1]: use interior points slightly; fine since k smooth; check ratio anyway
    print(u, ratio(Z+u*H,Z))
