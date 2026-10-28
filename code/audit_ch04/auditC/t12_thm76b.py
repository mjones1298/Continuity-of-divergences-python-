import numpy as np
from lib import *
from diamond import nu_est
rng=np.random.default_rng(8)
def hBSv(t):
    t=np.asarray(t,float); return np.where(t>0, t*np.log(np.where(t>0,t,1)/(1-np.clip(t,0,1-1e-300))),0.0)
from scipy.optimize import minimize_scalar
mf=minimize_scalar(lambda t:float(hBSv(t)),bounds=(0,1),method='bounded').fun
print('m_f',mf)
def hJ(t):
    t=np.asarray(t,float); return (1-t)*(t/(1-t)-1)*np.log(t/(1-t))
mfJ=minimize_scalar(lambda t:float(hJ(t)),bounds=(1e-9,1-1e-9),method='bounded').fun
worst={'BS_lb':1e9,'BS_est':1e9,'J_lb':1e9,'J_est':1e9}
for trial in range(400):
    dX=rng.integers(1,4); dY=rng.integers(1,4); r=rng.integers(1,min(dX*dY,3)+1)
    Ks=rand_kraus(dX,dY,r)
    if r>1 and rng.random()<0.5: Ks[-1]=Ks[0]*rng.normal()+10**rng.uniform(-3,0)*Ks[-1]
    W=W_of(Ks)
    if np.linalg.matrix_rank(W,tol=1e-13)<r: continue
    J=W@W.conj().T; ev=np.linalg.eigvalsh(J); lm=ev[ev>1e-12*ev.max()].min()
    nu_lb2=lm/min(dX,dY)
    nu_e2=nu_est(Ks,restarts=6)**2
    on=np.linalg.norm(sum(K.conj().T@K for K in Ks),2)
    spec=rng.uniform(0.001,0.999,size=r)
    T=rand_T(r,spec=spec)
    for name,h,m in (('BS',hBSv,mf),('J',hJ,mfJ)):
        D=np.linalg.eigvalsh(M_of(W,T,h,dY,dX))[-1]
        for lam in spec:
            for tag,n2 in (('lb',nu_lb2),('est',nu_e2)):
                rhs=n2*(float(h(lam))-m)+min(m,0)*on
                worst[name+'_'+tag]=min(worst[name+'_'+tag],D-rhs)
print(worst)
