# P3: second-order hierarchy for the qubit family N_{p,delta}=D_p o R_X(delta) vs N_p (U=1).
import numpy as np, mpmath as mp, sys, scipy.optimize as so
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))  # glib.py in code/common
from glib import *
mp.mp.dps=40
X=np.array([[0,1],[1,0]],complex)
def Rx(d): return np.cos(d/2)*np.eye(2)-1j*np.sin(d/2)*X
def Jn(p,d): return (1-p)*choi([Rx(d)])+p/2*np.eye(4)
def relent(r,s):
    return np.real(np.trace(r@(fn(r,lambda w: np.log(np.clip(w,1e-300,None)))-fn(s,np.log))))
def rootfid(r,s):
    a=sqrtm(r); return np.real(np.sum(np.sqrt(np.clip(np.linalg.eigvalsh(herm(a@s@a)),0,None))))
def out_state(J,tau):
    # input pure state with X-marginal tau: output (M x 1)... use (1_Y x sqrt(tau)^T) J (1_Y x sqrt(tau)^T)^*  (J on Y x X)
    M=np.kron(np.eye(2),sqrtm(tau).T); return herm(M@J@M.conj().T)
# exact coefficients
for p in [1e-1,1e-2,1e-3,1e-4]:
    IR=2*(1-p)**2*(2-p)/(p*(4-3*p)); FB=(1-p)/2*np.log((4-3*p)/p); FS=2*(1-p)**2/(2-p)
    d=1e-4*min(1,np.sqrt(p))
    J0=Jn(p,0); J1=Jn(p,d)
    oneF=1-Fhat(J1,J0,2,2); Dbs=np.linalg.eigvalsh(trY(herm(sqrtm(J1)@fn(sqrtm(J1)@np.linalg.inv(J0)@sqrtm(J1),np.log)@sqrtm(J1)),2,2)).max()
    U=relent(J1/2,J0/2); oneRF=1-rootfid(J1/2,J0/2)
    print(f'p={p:g}: (1-F^)/(d^2 I_RLD/8)={oneF/(d*d*IR/8):.6f}  D_BS/(d^2 I_RLD/2)={Dbs/(d*d*IR/2):.6f}  D_Choi/(d^2 F_BKM/2)={U/(d*d*FB/2):.6f}  (1-sqrtF_Choi)/(d^2 F_SLD/8)={oneRF/(d*d*FS/8):.6f}')
    print(f'      p*I_RLD={p*IR:.5f}  F_BKM={FB:.5f} (0.5 ln(4/p)={0.5*np.log(4/p):.5f})  F_SLD={FS:.5f}')
# closed forms at finite delta
for p,d in [(0.01,0.3),(0.001,0.05),(0.2,1.2)]:
    J0=Jn(p,0); J1=Jn(p,d)
    print(f'p={p},delta={d}: D_Choi={relent(J1/2,J0/2):.12f} closed={(1-p)*np.sin(d/2)**2*np.log((4-3*p)/p):.12f};  sqrtF_Choi={rootfid(J1/2,J0/2):.12f} closed={p/2+np.sqrt((1-p/2)**2-(1-p)**2*np.sin(d/2)**2):.12f}')
# channel-level: optimise input marginal tau (qubit reference suffices for pure inputs)
def tau_from(x):
    v=np.array([x[0]+1j*x[1],x[2]+1j*x[3]]); a=np.outer(v,v.conj())+np.diag([x[4]**2,x[5]**2]); return herm(a/np.trace(a))
rng=np.random.default_rng(0)
for p,d in [(0.01,0.05),(0.001,0.01),(0.1,0.3)]:
    J0=Jn(p,0); J1=Jn(p,d)
    me=np.eye(2)/2
    bestU=relent(out_state(J1,me),out_state(J0,me)); bestF=rootfid(out_state(J1,me),out_state(J0,me))
    for r in range(20):
        x0=rng.normal(size=6)
        rU=so.minimize(lambda x: -relent(out_state(J1,tau_from(x)),out_state(J0,tau_from(x))),x0,method='Nelder-Mead',options={'maxiter':4000})
        rF=so.minimize(lambda x: rootfid(out_state(J1,tau_from(x)),out_state(J0,tau_from(x))),x0,method='Nelder-Mead',options={'maxiter':4000})
        bestU=max(bestU,-rU.fun); bestF=min(bestF,rF.fun)
    print(f'p={p},delta={d}: Umegaki channel/Choi = {bestU/relent(out_state(J1,me),out_state(J0,me)):.6f};  (1-sqrtF) channel/Choi = {(1-bestF)/(1-rootfid(out_state(J1,me),out_state(J0,me))):.6f}')
