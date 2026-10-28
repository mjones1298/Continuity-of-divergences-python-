# 50-digit re-evaluation of the p2 objective at a saved Phi (x*): exact gradient (no threshold) + SDP,
# and one-sided mp slopes along the double-precision optimal direction.
import numpy as np, mpmath as mp, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))  # glib.py in code/common
mp.mp.dps=50
exec(open(Path(__file__).resolve().parent / 'p2_common.py').read())   # JDp, trYcp, grad_op (double, thresholded), best_dir
def phi_kraus(x):
    Gm=(x[:16]+1j*x[16:]).reshape(8,2); Q,_=np.linalg.qr(Gm); return [Q[2*k:2*k+2] for k in range(4)]
def tomp(M): return mp.matrix([[mp.mpc(complex(M[i,j])) for j in range(M.shape[1])] for i in range(M.shape[0])])
def mchoi(Ks):
    J=mp.matrix(4,4)
    for K in Ks:
        v=mp.matrix([K[0,0],K[0,1],K[1,0],K[1,1]]); J+=v*v.H
    return J
def mherm(A): return (A+A.H)/2
def fnm(A,f):
    w,U=mp.eighe(mherm(A)); return U*mp.diag([f(x) for x in w])*U.H
def mgm(B,A):   # B # A, B>0, exact sqrt (clip only |x|<1e-45)
    Bh=fnm(B,mp.sqrt); Bih=fnm(B,lambda x:1/mp.sqrt(x)); M=Bih*A*Bih
    return Bh*fnm(M,lambda x: mp.sqrt(x) if x>0 else mp.mpf(0))*Bh
def mtrY(M):
    T=mp.matrix(2,2)
    for i in range(2):
        for a in range(2):
            for b in range(2): T[a,b]+=M[i*2+a,i*2+b]
    return T
def F(A,B): return min(mp.eighe(mherm(mtrY(mgm(B,A))))[0])
def mp_grad(A,B,h):
    n=4; G=np.zeros((4,4),complex); basis=[]
    for i in range(n):
        for j in range(i,n):
            if i==j: E=np.zeros((n,n),complex); E[i,i]=1; basis.append(E)
            else:
                E=np.zeros((n,n),complex); E[i,j]=E[j,i]=1/np.sqrt(2); basis.append(E)
                E=np.zeros((n,n),complex); E[i,j]=1j/np.sqrt(2); E[j,i]=-1j/np.sqrt(2); basis.append(E)
    for E in basis:
        Em=tomp(E); g=(F(A,B+h*Em)-F(A,B-h*Em))/(2*h); G+=complex(g)*E
    return G
def check(x,p,label):
    Ks=phi_kraus(x); A=mchoi([tomp(K) for K in Ks]); B=tomp(JDp(p))
    eA=[float(mp.re(e)) for e in mp.eighe(mherm(A))[0]]
    Gd=grad_op(choi(Ks),JDp(p),1e-9); vd=p*best_dir(Gd)
    h=mp.mpf('1e-25'); Gm=mp_grad(A,B,h); vm=p*best_dir(Gm)
    # one-sided slopes along the double-precision optimal direction
    import cvxpy as cp
    n=4; E=cp.Variable((n,n),hermitian=True); Y0=cp.Variable((n,n),hermitian=True); Y1=cp.Variable((n,n),hermitian=True); t0=cp.Variable(); t1=cp.Variable()
    cons=[cp.bmat([[Y0,-E],[-E,Y1]])>>0, trYcp(Y0)<<t0*np.eye(2), trYcp(Y1)<<t1*np.eye(2), (t0+t1)/2<=1, trYcp(E)==0]
    cp.Problem(cp.Maximize(cp.real(cp.trace(E@Gd))),cons).solve(solver='CLARABEL'); Ed=tomp(E.value)
    F0=F(A,B); hs=mp.mpf('1e-20')
    sp=(F(A,B+hs*Ed)-F0)/hs; sm=(F0-F(A,B-hs*Ed))/hs
    print(f'{label}: Choi eigs of Phi = {["%.3e"%e for e in eA]}, Fhat={float(F0):.6f}')
    print(f'   double (thresholded) p*sup = {vd:.6f};  mp-50 exact-gradient p*sup = {vm:.6f}')
    print(f'   mp one-sided slopes along double-optimal E: p*dF/de(+) = {float(sp)*p:.6f}, p*dF/de(-) = {float(sm)*p:.6f}',flush=True)
if __name__=='__main__':
    p=float(sys.argv[2]); x=np.load(sys.argv[1]); check(x,p,sys.argv[1])
