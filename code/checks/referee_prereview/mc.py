import numpy as np, cvxpy as cp
from scipy.optimize import minimize
rng=np.random.default_rng(1)
d=2
def hpow(A,a):
    w,V=np.linalg.eigh((A+A.conj().T)/2); w=np.clip(w,1e-300,None); return (V*w**a)@V.conj().T
def hlog(A):
    w,V=np.linalg.eigh((A+A.conj().T)/2); w=np.clip(w,1e-300,None); return (V*np.log(w))@V.conj().T
def ptrB(X): return np.einsum('ajbj->ab',X.reshape(d,d,d,d))
def choi(K): # unnormalized, Tr_B J = I
    J=np.zeros((d*d,d*d),complex)
    for k in K:
        v=np.zeros(d*d,complex)
        for i in range(d):
            e=np.zeros(d);e[i]=1; v+=np.kron(e,k@e)
        J+=np.outer(v,v.conj())
    return J
def ad(g): return [np.array([[1,0],[0,np.sqrt(1-g)]]),np.array([[0,np.sqrt(g)],[0,0]])]
def dep(J,p): return (1-p)*J+p*np.eye(4)/2
def randchoi():
    G=rng.normal(size=(8,2))+1j*rng.normal(size=(8,2)); Q,_=np.linalg.qr(G) # isometry 2->8
    K=[Q[2*i:2*i+2,:] for i in range(4)]; return choi(K)
def geo_chan_fid(J1,J2):  # root geometric channel fidelity = lambda_min Tr_B(J1#J2)
    Ah=hpow(J1,.5);Aih=hpow(J1,-.5); G=Ah@hpow(Aih@J2@Aih,.5)@Ah
    return np.min(np.linalg.eigvalsh(ptrB(G)))
def bs_chan(J1,J2):
    Ah=hpow(J1,.5); X=Ah@hpow(J2,-1)@Ah; P=Ah@hlog(X)@Ah
    return np.max(np.linalg.eigvalsh(ptrB(P)))
def umeg_chan(J1,J2):
    def f(x):
        th,ph=x; r=np.array([[np.cos(th/2)**2, np.cos(th/2)*np.sin(th/2)*np.exp(-1j*ph)],[np.cos(th/2)*np.sin(th/2)*np.exp(1j*ph),np.sin(th/2)**2]])
        r=0.999999*r+1e-6*np.eye(2)/2
        R=np.kron(hpow(r.T,.5),np.eye(2)); A=R@J1@R; B=R@J2@R
        return -np.real(np.trace(A@(hlog(A)-hlog(B))))
    best=min((minimize(f,x0,method='Nelder-Mead') for x0 in [(0.3,0),(1.5,0),(2.8,1),(1.5,2)]),key=lambda r:r.fun)
    return -best.fun
def hs_chan(J1,J2,g):
    Q=cp.Variable((4,4),hermitian=True); r=cp.Variable((2,2),hermitian=True)
    X=J1-g*J2
    cons=[Q>>0, cp.kron(r,np.eye(2))-Q>>0, cp.real(cp.trace(r))==1]
    pr=cp.Problem(cp.Maximize(cp.real(cp.trace(Q@X))),cons); pr.solve(solver=cp.SCS,eps=1e-7)
    return pr.value
gN,gM=0.90,0.99
print("N=AD(0.90), M=AD(0.99), both + depolarizing floor p; tomography error eps: J->(1-eps)J+eps*J_rand (diamond <= 2 eps)")
print("p, eps : mean+-std over 30 draws for  -log F_geo^2 | D_BS | D_Umegaki | E_{gamma=2} ; true values first")
for p in [1e-2,1e-3,1e-4]:
    JN=dep(choi(ad(gN)),p); JM=dep(choi(ad(gM)),p)
    tv=[-2*np.log(geo_chan_fid(JN,JM)), bs_chan(JN,JM), umeg_chan(JN,JM), hs_chan(JN,JM,2.0)]
    print(f"p={p:.0e} TRUE: geo={tv[0]:.3f} BS={tv[1]:.3f} Um={tv[2]:.3f} HS={tv[3]:.4f}")
    for eps in [1e-3,1e-2]:
        vals=[]
        for _ in range(30):
            Jn=(1-eps)*JN+eps*randchoi(); Jm=(1-eps)*JM+eps*randchoi()
            vals.append([-2*np.log(geo_chan_fid(Jn,Jm)), bs_chan(Jn,Jm), umeg_chan(Jn,Jm), hs_chan(Jn,Jm,2.0)])
        v=np.array(vals); m=v.mean(0); s=v.std(0); mx=np.abs(v-np.array(tv)).max(0)
        print(f"   eps={eps:.0e}: geo {m[0]:.3f}+-{s[0]:.3f} (maxdev {mx[0]:.3f}) | BS {m[1]:.3f}+-{s[1]:.3f} (maxdev {mx[1]:.3f}) | Um {m[2]:.3f}+-{s[2]:.3f} (maxdev {mx[2]:.3f}) | HS {m[3]:.4f}+-{s[3]:.4f} (maxdev {mx[3]:.4f})")
