# P1: n-qubit local depolarizing; lambda_min; closed form of F-hat, D_BS, C-hat for Pauli-string miscalibration.
import numpy as np, itertools, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))  # glib.py in code/common
from glib import *
from functools import reduce
I2=np.eye(2); X=np.array([[0,1],[1,0]],complex); Y=np.array([[0,-1j],[1j,0]]); Z=np.diag([1.,-1]).astype(complex)
P1=[I2,X,Y,Z]
def kron(*a): return reduce(np.kron,a)
def paulis(n): return [(lab,kron(*[P1[i] for i in lab])) for lab in itertools.product(range(4),repeat=n)]
def local_dep_probs(n,p):
    q1=[1-3*p/4,p/4,p/4,p/4]
    return {lab:np.prod([q1[i] for i in lab]) for lab in itertools.product(range(4),repeat=n)}
def choi_pauli(n,q,V):
    # channel rho -> sum_Q q_Q Q V rho V^* Q
    return choi([np.sqrt(q[lab])*Q@V for lab,Q in paulis(n)])
def expmP(P,delta): return np.cos(delta/2)*np.eye(len(P))-1j*np.sin(delta/2)*P
def gmean_s(A,B,s):  # A #_s B for A>0
    return gmean_pd(A,B,s)
def DBS(A,B,dY,dX):
    Ah=sqrtm(A); M=Ah@np.linalg.inv(B)@Ah
    L=fn(M,np.log); return np.linalg.eigvalsh(trY(herm(Ah@L@Ah),dY,dX))
rng=np.random.default_rng(3)
# lambda_min
for n in [1,2,3]:
    for p in [0.1,0.01]:
        d=2**n; U=rand_unitary(rng,d)
        J=choi_pauli(n,local_dep_probs(n,p),U)
        Jg=(1-p)*choi([U])+p*np.eye(d*d)/d
        print(f'n={n} p={p}: local lam_min={np.linalg.eigvalsh(J).min():.6e} vs (p/2)^n={(p/2)**n:.6e};  global lam_min={np.linalg.eigvalsh(Jg).min():.6e} vs p/d={p/d:.6e}')
def phi(x): return np.sqrt(x/(1+x))*np.arcsinh(np.sqrt(x))
# closed forms for n=2
n=2; d=4
for (p,delta,Plab) in [(0.05,0.3,(1,1)),(0.01,0.05,(1,0)),(0.01,0.02,(3,1)),(0.2,1.0,(2,3)),(0.003,0.004,(1,2))]:
    q=local_dep_probs(n,p); P=kron(*[P1[i] for i in Plab]); U=rand_unitary(rng,d)
    J0=choi_pauli(n,q,U); J1=choi_pauli(n,q,expmP(P,delta)@U)
    G=gmean_pd(J0,J1); TG=trY(G,d,d)
    F_direct=np.linalg.eigvalsh(TG).min(); prop=np.abs(TG-np.trace(TG)/d*np.eye(d)).max()
    # closed form: pair Q with QP
    labs=list(q.keys()); done=set(); F_cf=0; D_cf=0
    for lab in labs:
        if lab in done: continue
        Q=kron(*[P1[i] for i in lab]); QP=Q@P
        # find label of QP up to phase
        for lab2,Q2 in paulis(n):
            ov=np.trace(Q2.conj().T@QP)/d
            if abs(abs(ov)-1)<1e-9: break
        done|={lab,lab2}
        a,b=q[lab],q[lab2]
        x=(a-b)**2*np.sin(delta/2)**2/(4*a*b)
        F_cf+=(a+b)/np.sqrt(1+x); D_cf+=2*(a+b)*phi(x)
    dbs=DBS(J1,J0,d,d)
    # C-hat: scan s
    Qs=[np.linalg.eigvalsh(trY(gmean_pd(J0,J1,s),d,d)).min() for s in np.linspace(0.02,0.98,49)]
    print(f'p={p} delta={delta} P={Plab}: F direct={F_direct:.12f} closed={F_cf:.12f} |TrY - c1|={prop:.1e}; '
          f'D_BS max={dbs.max():.10f} min={dbs.min():.10f} closed={D_cf:.10f}; min_s Q_s={min(Qs):.10f} (argmin s={np.linspace(0.02,0.98,49)[np.argmin(Qs)]:.2f})')
