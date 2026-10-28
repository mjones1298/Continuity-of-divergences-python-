import mpmath as mp
import numpy as np
mp.mp.dps = 50

def tomp(A):
    A=np.asarray(A)
    M=mp.matrix(A.shape[0],A.shape[1])
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            M[i,j]=mp.mpc(complex(A[i,j]).real, complex(A[i,j]).imag)
    return M

def mfun(A, f):
    # A hermitian mp matrix
    n=A.rows
    H=(A+A.H)/2
    E,Q=mp.eigh(H)
    D=mp.diag([f(E[i]) for i in range(n)])
    return Q*D*Q.H

def persp_trace(A,B,f,eta):
    n=A.rows
    I=mp.eye(n)
    A=A+eta*I; B=B+eta*I
    Bh=mfun(B,lambda x: mp.sqrt(x)); Bih=mfun(B,lambda x: 1/mp.sqrt(x))
    X=Bih*A*Bih
    FX=mfun(X,f)
    P=Bh*FX*Bh
    return mp.re(sum(P[i,i] for i in range(n)))

def Df_mp(A,B,f,etas=(mp.mpf('1e-20'),)):
    return [persp_trace(A,B,f,e) for e in etas]
