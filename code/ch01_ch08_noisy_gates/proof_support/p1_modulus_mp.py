# high-precision re-check of n-qubit modulus claims (mpmath, 40 digits)
import mpmath as mp, itertools, numpy as np
mp.mp.dps=40
I2=mp.eye(2); X=mp.matrix([[0,1],[1,0]]); Y=mp.matrix([[0,-1j],[1j,0]]); Z=mp.matrix([[1,0],[0,-1]])
P1=[I2,X,Y,Z]
def kron(A,B):
    m,n=A.rows,A.cols; r,c=B.rows,B.cols; K=mp.matrix(m*r,n*c)
    for i in range(m):
        for j in range(n):
            for k in range(r):
                for l in range(c): K[i*r+k,j*c+l]=A[i,j]*B[k,l]
    return K
def kr(labs):
    M=P1[labs[0]]
    for l in labs[1:]: M=kron(M,P1[l])
    return M
def vec(V): return mp.matrix([V[i,j] for i in range(V.rows) for j in range(V.cols)])
def choi(Ks):
    n=Ks[0].rows*Ks[0].cols; J=mp.matrix(n,n)
    for K in Ks:
        v=vec(K); J+=v*v.H
    return J
def herm(A): return (A+A.H)/2
def fnm(A,f):
    w,U=mp.eighe(herm(A)); return U*mp.diag([f(x) for x in w])*U.H
def gmean(B,A):  # B # A with B>0
    Bh=fnm(B,mp.sqrt); Bih=fnm(B,lambda x:1/mp.sqrt(x))
    return Bh*fnm(Bih*A*Bih,lambda x: mp.sqrt(max(x,0)))*Bh
def trY(M,d):
    T=mp.matrix(d,d)
    for i in range(d):
        for a in range(d):
            for b in range(d): T[a,b]+=M[i*d+a,i*d+b]
    return T
def lmin(M): return min(mp.eighe(herm(M))[0])
def F(A,B,d): return lmin(trY(gmean(B,A),d))
for n,Plab in [(1,(1,)),(2,(1,3))]:
    d=2**n; P=kr(Plab); Id=mp.eye(d)
    for p in [mp.mpf('1e-2'),mp.mpf('1e-3')]:
        q1=[1-3*p/4,p/4,p/4,p/4]
        Ks=[mp.sqrt(np.prod([q1[i] for i in lab]))*kr(lab) for lab in itertools.product(range(4),repeat=n)]
        B=choi(Ks); lam=min(mp.eighe(B)[0])
        E=(choi([P])-choi([Id]))/2
        h=mp.mpf('1e-15')*lam
        A=choi([P]); r=(mp.log(F(A,B+h*E,d))-mp.log(F(A,B-h*E,d)))/(2*h)
        print(f'n={n} p={float(p)}: lam={float(lam):.4e}  dlogF/de (Phi=Ad_P) * 4lam/d = {float(r*4*lam/d):.10f}',flush=True)
        best=0
        for t in np.linspace(0.8,2.2,15):
            th=mp.sqrt(lam/d)*t; V=mp.cos(th)*Id-1j*mp.sin(th)*P; A=choi([V])
            g=(F(A,B+h*E,d)-F(A,B-h*E,d))/(2*h); best=max(best,g)
        print(f'      max_th dF/de * lam/d = {float(best*lam/d):.6f}   (p->0 limit 1/(6 sqrt3)={1/(6*np.sqrt(3)):.6f})',flush=True)
