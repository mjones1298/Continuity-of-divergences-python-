import numpy as np, sys
sys.path.insert(0,'.')
from lib import *
rng = np.random.default_rng(5)
dY, dX = 2, 2; N=4
KP = random_kraus(dY, dX, 4, rng); KQ = random_kraus(dY, dX, 4, rng)
JP = sum(np.outer(vec(K), vec(K).conj()) for K in KP); JQ = sum(np.outer(vec(K), vec(K).conj()) for K in KQ)
S = JP+JQ
w,V = np.linalg.eigh(S); Wf = V*np.sqrt(w); Wi = V/np.sqrt(w)
T = herm(Wi.conj().T@JP@Wi)
fs = {
 'Hellinger': (lambda t: (1-np.sqrt(t))**2, 1.0, 1.0),
 'LC': (lambda t: (t-1)**2/(t+1), 1.0, 1.0),
 'JS': (lambda t: t*np.log(t)-(1+t)*np.log((1+t)/2), np.log(2), np.log(2)),
 'f=-sqrt(t)+2t': (lambda t: -np.sqrt(t)+2*t if False else 1-np.sqrt(t), 1.0, 0.0),   # example f(0+)=1,f'(inf)=0 (neg sqrt + 1)
}
def hf(f, f0, finf):
    def h(t):
        out = np.empty_like(t)
        for i,x in enumerate(t):
            if x<=1e-15: out[i]=f0
            elif x>=1-1e-15: out[i]=finf
            else: out[i]=(1-x)*f(x/(1-x))
        return out
    return h
def tensor_apply(Ks, rho):
    return sum(np.kron(K, np.eye(dX)) @ rho @ np.kron(K, np.eye(dX)).conj().T for K in Ks)
for name,(f,f0,finf) in fs.items():
    h = hf(f,f0,finf)
    wt,U = np.linalg.eigh(T)
    Mf = ptrace_Y(Wf@((U*h(wt))@U.conj().T)@Wf.conj().T, dY, dX)
    lm = np.linalg.eigvalsh(herm(Mf))[-1]
    worst=0; mx=-9
    for s in range(100):
        A = rng.normal(size=(dX,dX))+1j*rng.normal(size=(dX,dX)); A/=np.linalg.norm(A)
        u=vec(A); rho=np.outer(u,u.conj())
        P=tensor_apply(KP,rho); Q=tensor_apply(KQ,rho)
        Qh=psd_fun(Q,np.sqrt); Qih=psd_fun(Q,lambda x:1/np.sqrt(x))
        Mx=herm(Qih@P@Qih); wm,Um=np.linalg.eigh(Mx)
        val=np.trace(Qh@((Um*f(wm))@Um.conj().T)@Qh).real
        pred=np.trace((A@A.conj().T).T@Mf).real
        worst=max(worst,abs(val-pred)); mx=max(mx,val)
    print("%-14s max|D_f(rho) - tr((AA*)^T M_f)|=%.2e   max sampled=%.5f <= lambda_max(M_f)=%.5f" % (name, worst, mx, lm))
