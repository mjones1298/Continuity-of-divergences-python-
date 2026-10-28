import sys; sys.path.insert(0,'.')
from lib import *
from scipy.optimize import minimize
def pw(E,p):
    w,V=eigh(herm(E)); w=np.clip(w,0,None); return (V*(w**p))@V.conj().T
def h(E,a):
    w,V=eigh(herm(E)); w=np.clip(w,0,1); return (V*(w**a*(1-w)**(1-a)))@V.conj().T
def param(x,n):
    A=(x[:n*n]+1j*x[n*n:2*n*n]).reshape(n,n); B=(x[2*n*n:3*n*n]+1j*x[3*n*n:4*n*n]).reshape(n,n)
    # E = spectral squashing of hermitian A into [0,1] via sigmoid
    def sq(H):
        H=herm(H); w,V=eigh(H); return (V*(1/(1+np.exp(-w))))@V.conj().T
    return sq(A),sq(B)
def ratio(x,n,a,delta_target=None):
    E,F=param(x,n)
    d=opnorm(E-F)
    if d<1e-12: return 0
    return opnorm(h(E,a)-h(F,a))/(d**a+d**(1-a))
for a in [0.2,0.5,0.8]:
    best=0
    for n in [1,2,3]:
        for s in range(15):
            x0=rng.normal(size=4*n*n)*3
            res=minimize(lambda x:-ratio(x,n,a),x0,method='Nelder-Mead',options={'maxiter':3000,'xatol':1e-8,'fatol':1e-10})
            best=max(best,-res.fun)
    print("alpha",a,"best ratio h-diff/(d^a+d^(1-a))",best)
# true sup over d of ||h(E)-h(E')||/d^{min}: scalar case, E'=0, E=d
for a in [0.2,0.5,0.8]:
    d=1e-8; print(a,(d**a*(1-d)**(1-a))/ (d**min(a,1-a)))
