# Numerical checks: Example 10.6 (tau/delta -> 2*sqrt2, PSD), Lemma 10.5 random bound, Thm 9.6(d) Lip(k_z)->sqrt2
import numpy as np
from scipy.linalg import sqrtm, eigvalsh
rng=np.random.default_rng(1)
Z=np.diag([1.,-1.]); X=np.array([[0,1],[1,0.]])
U0=(Z+X)/np.sqrt(2); U1=(Z-X)/np.sqrt(2)
def inv_sqrt(C): 
    w,v=np.linalg.eigh(C); return (v/np.sqrt(w))@v.conj().T
def ex(delta):
    T0=np.diag([delta,1-delta]); I=np.eye(2)
    A=[0.5*(T0+s*delta*U0) for s in (1,-1)]
    B=[0.5*(I-T0+s*delta*U1) for s in (1,-1)]
    mins=min(eigvalsh(M).min() for M in A+B)
    C=[A[i]+B[i] for i in range(2)]
    assert np.allclose(C[0]+C[1],I)
    T=[]
    for i in range(2):
        w,v=np.linalg.eigh(C[i]); ai=(v*np.sqrt(w))@v.T; aiinv=(v/np.sqrt(w))@v.T
        T.append(aiinv@A[i]@aiinv)
    tau=np.linalg.norm(T[0]-T[1],2)
    d=max(np.linalg.norm(A[0]-A[1],2),np.linalg.norm(B[0]-B[1],2))
    gam=np.linalg.norm(C[0]-0.5*I,2)
    bound=2*np.sqrt(2)*d/np.sqrt(1-4*gam**2)
    return tau/d, mins, d, tau<=bound+1e-12, 2*d/(1-2*d)
for dl in [0.2,0.1,0.05,0.01,0.001,1e-4]:
    r,m,d,ok,comm=ex(dl); print(f"delta={dl}: tau/delta={r:.6f} minEig={m:.3e} delta_meas={d:.6f} lemma10.5 holds={ok} commuting bound 2d/(1-2d)={comm:.6f}")
print("2*sqrt2=",2*np.sqrt(2))
# random test of Lemma 10.5: random A_i,B_i PSD with A1+B1+A2+B2=I
worst=0;cnt=0
for _ in range(20000):
    r=rng.integers(1,5)
    G=[rng.normal(size=(r,r))+1j*rng.normal(size=(r,r)) for _ in range(4)]
    P=[g@g.conj().T for g in G]; S=sum(P); Si=inv_sqrt(S)
    A1,B1,A2,B2=[Si@p@Si for p in P]
    # make near each other: perturb? just use random plus mixing
    t=rng.uniform(0,1)**3
    A2=(1-t)*A1+t*A2; B2=(1-t)*B1+t*B2
    # renormalise so that C1+C2 = I: C1=A1+B1, C2=A2+B2 -> need sum I: rescale via common congruence
    S=A1+B1+A2+B2; Si=inv_sqrt(S)
    A1,B1,A2,B2=[Si@p@Si for p in (A1,B1,A2,B2)]
    C1=A1+B1;C2=A2+B2
    d=max(np.linalg.norm(A1-A2,2),np.linalg.norm(B1-B2,2)); 
    gam=np.linalg.norm(C1-0.5*np.eye(r),2)
    if d>=0.5 or 2*gam>=1: continue
    T=[]
    for C,A in ((C1,A1),(C2,A2)):
        w,v=np.linalg.eigh(C); aiinv=(v/np.sqrt(w))@v.conj().T; T.append(aiinv@A@aiinv)
    tau=np.linalg.norm(T[0]-T[1],2)
    bound=2*np.sqrt(2)*d/np.sqrt(1-4*gam**2)
    worst=max(worst,tau/bound);cnt+=1
print("Lemma 10.5 random: tested",cnt,"max tau/bound =",worst)
# Thm 9.6(d): k_z(t)=t+2 z^2/(t+z) : max|k'| on [0,1] =1, Lip^{[0,1]} -> sqrt2 as z->0 (two-point Schur norm)
def lip_two(g,t1,t2,n=1):
    # Lipschitz via operator-norm of divided difference matrix for 2 points: [[g'(t1), g1],[g1, g'(t2)]]
    pass
import itertools
def kz(z): return (lambda t: t+2*z*z/(t+z)), (lambda t: 1-2*z*z/(t+z)**2)
for z in [0.1,0.01,0.001]:
    k,kp=kz(z)
    ts=np.linspace(0,1,2001); print(z,"max|k'|=",np.abs(kp(ts)).max())
    # Schur multiplier norm of 2x2 divided-difference matrix [[k'(a), k[a,b]],[k[a,b],k'(b)]] over a<b grid (lower bound for Lip)
    best=0
    for a in [0,z/4,z/2,z,2*z]:
        for b in [1,0.5,0.1,0.02]:
            if b<=a: continue
            k1=(k(b)-k(a))/(b-a)
            M=np.array([[kp(a),k1],[k1,kp(b)]])
            # Schur multiplier norm of 2x2 real symmetric matrix: SDP-free formula via min over factorisations; use brute force with cvxpy if available
            import cvxpy as cp
            Y1=cp.Variable((2,2),symmetric=True);Y2=cp.Variable((2,2),symmetric=True);t=cp.Variable()
            pr=cp.Problem(cp.Minimize(t),[cp.bmat([[Y1,M],[M.T,Y2]])>>0,cp.diag(Y1)<=t,cp.diag(Y2)<=t]); pr.solve(solver=cp.CLARABEL)
            best=max(best,t.value)
    print("  z=",z,"best Schur norm over sampled pairs =",best,"(sqrt2=%.6f)"%np.sqrt(2))
