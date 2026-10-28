import mpmath as mp
mp.mp.dps = 90
def mfun(A, f):
    E, Q = mp.eigsy(A); n=A.rows
    return Q*mp.diag([f(E[i]) for i in range(n)])*Q.T
def Dreg(A,B,f,eta):
    n=A.rows; Ae=A+eta*mp.eye(n); Be=B+eta*mp.eye(n)
    Bh=mfun(Be,mp.sqrt); Bih=mfun(Be,lambda x:1/mp.sqrt(x))
    X=Bih*Ae*Bih; X=(X+X.T)/2
    P=Bh*mfun(X,f)*Bh
    return sum(P[i,i] for i in range(n))
def kron(A,B):
    ra,ca,rb,cb=A.rows,A.cols,B.rows,B.cols
    K=mp.zeros(ra*rb,ca*cb)
    for i in range(ra):
        for j in range(ca):
            for k in range(rb):
                for l in range(cb):
                    K[i*rb+k,j*cb+l]=A[i,j]*B[k,l]
    return K
mp.mp.dps=90
dX,dY=2,2
import random
random.seed(3)
def rnd(): return mp.mpf(random.uniform(-1,1))
V1=mp.matrix(dY,dX); V2=mp.matrix(dY,dX); Wd=mp.matrix(dY,dX)
for i in range(dY):
    for j in range(dX):
        V1[i,j]=rnd(); Wd[i,j]=rnd()
def vec(K):
    return mp.matrix([K[i,j] for i in range(K.rows) for j in range(K.cols)])
flog=lambda t: t*mp.log(t) if t>0 else mp.mpf(0)
h=lambda t: t*mp.log(t/(1-t)) if t>0 else mp.mpf(0)
for tau in [mp.mpf('1e-3'),mp.mpf('1e-6'),mp.mpf('1e-10')]:
    V2=V1+tau*Wd
    W=mp.matrix(dY*dX,2)
    for i,v in enumerate((vec(V1),vec(V2))):
        for k in range(dY*dX): W[k,i]=v[k]
    th=mp.mpf('0.9'); Q=mp.matrix([[mp.cos(th),-mp.sin(th)],[mp.sin(th),mp.cos(th)]])
    T=Q*mp.diag([mp.mpf('0.7'),mp.mpf('0.2')])*Q.T
    J1=W*T*W.T; J2=W*(mp.eye(2)-T)*W.T
    A=mp.matrix([[mp.mpf('0.8'),mp.mpf('0.3')],[mp.mpf('-0.2'),mp.mpf('0.5')]]); A=A/mp.sqrt(sum(A[i,j]**2 for i in range(2) for j in range(2)))
    C=kron(mp.eye(dY),A.T)
    P1=C*J1*C.T; P2=C*J2*C.T
    # prediction
    H=mfun(T,h)
    Wt=C*W
    pred=sum((Wt*H*Wt.T)[i,i] for i in range(dY*dX))
    # eigenvalues of Choi
    E,_=mp.eigsy(J1+J2)
    vals=[Dreg(P1,P2,flog,mp.mpf(10)**(-k)) for k in (40,60)]
    print('tau',mp.nstr(tau,3),'min eig J(Omega)',mp.nstr(min(E),5),'pred',mp.nstr(pred,18),'reg',[mp.nstr(v,18) for v in vals])
