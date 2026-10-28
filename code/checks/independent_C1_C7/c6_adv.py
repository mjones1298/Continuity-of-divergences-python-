from pathlib import Path
import numpy as np, sys
from scipy.optimize import minimize
from qlib import *
import importlib.util
spec=importlib.util.spec_from_file_location('c6m',Path(__file__).resolve().parent / 'c6.py')
src=open(Path(__file__).resolve().parent / 'c6.py').read().split('results=[]')[0]
ns={}; exec(src,ns)
bound_terms=ns['bound_terms']
rng=np.random.default_rng(int(sys.argv[1]))
def chan(x,r):
    G=(x[:4*r]+1j*x[4*r:8*r]).reshape(2*r,2); Q,R=np.linalg.qr(G); Q=Q*np.sign(np.diag(R).real+1e-300)
    return choi_from_kraus([Q[2*k:2*k+2] for k in range(r)])
def F(JA,JB):
    if min(np.linalg.eigvalsh(JA)[0],np.linalg.eigvalsh(JB)[0])>1e-7: return fid_np(JA,JB,2,2)
    return fid_sub(JA,JB,2,2)
mode=sys.argv[2]
def build(x):
    if mode=='full':   # four generic rank-r channels
        r=int(sys.argv[3]); n=8*r
        return [chan(x[i*n:(i+1)*n],r) for i in range(4)]
    if mode=='mix':    # Phi1,Psi1 rank r; Phi2,Psi2 = mixtures plus generic full-rank perturbation
        r=int(sys.argv[3]); n=8*r
        J1,K1=chan(x[:n],r),chan(x[n:2*n],r)
        E1,E2=chan(x[2*n:2*n+32],4),chan(x[2*n+32:2*n+64],4)
        a,b,c,d=1/(1+np.exp(-x[-4:]))*np.array([1,1,.05*(r==4),.05*(r==4)])
        J2=(1-a-c)*J1+a*K1+c*E1; K2=(1-b-d)*K1+b*J1+d*E2
        return [J1,K1,J2,K2]
def ratio(x):
    try:
        J1,K1,J2,K2=build(x)
        r1=np.linalg.matrix_rank(J1+K1,1e-8); r2=np.linalg.matrix_rank(J2+K2,1e-8); r3=np.linalg.matrix_rank(J1+K1+J2+K2,1e-8)
        if not (r1==r2==r3): return 0.0
        tau,th,nrm,c1,c2=bound_terms(J1,K1,J2,K2)
        if min(c1,c2)<1e-8: return 0.0
        lhs=abs(F(J1,K1)-F(J2,K2)); rhs=nrm*(np.sqrt(tau)+np.sqrt(2)*th)
        return lhs/rhs
    except Exception as e:
        return 0.0
best=(0,None)
for s in range(int(sys.argv[4])):
    if mode=='full': x0=rng.normal(size=4*8*int(sys.argv[3]))
    else:
        r=int(sys.argv[3]); x0=np.concatenate([rng.normal(size=16*r+64),rng.normal(size=4)*3-3])
    r_=minimize(lambda x:-ratio(x),x0,method='Nelder-Mead',options={'maxiter':3000,'maxfev':3000})
    if -r_.fun>best[0]: best=(-r_.fun,r_.x)
    print(f'start {s}: ratio={-r_.fun:.5f}',flush=True)
print('BEST',mode,sys.argv[3],best[0])
np.save(Path(__file__).resolve().parent / f'c6adv_{mode}_{sys.argv[3]}_{sys.argv[1]}.npy',best[1])
