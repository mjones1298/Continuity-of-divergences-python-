import mpmath as mp
mp.mp.dps = 80
def mfun(A, f):
    E, Q = mp.eigsy(A)   # real symmetric
    n = A.rows
    D = mp.diag([f(E[i]) for i in range(n)])
    return Q*D*Q.T
def Dreg(A,B,f,eta):
    n=A.rows
    Ae=A+eta*mp.eye(n); Be=B+eta*mp.eye(n)
    Bh=mfun(Be,mp.sqrt); Bih=mfun(Be,lambda x:1/mp.sqrt(x))
    X=Bih*Ae*Bih
    X=(X+X.T)/2
    P=Bh*mfun(X,f)*Bh
    return sum(P[i,i] for i in range(n))
flog=lambda t: t*mp.log(t) if t>0 else mp.mpf(0)
th=mp.mpf('0.7')
Q=mp.matrix([[mp.cos(th),-mp.sin(th)],[mp.sin(th),mp.cos(th)]])
for eps in [mp.mpf('1e-3'),mp.mpf('1e-6'),mp.mpf('1e-10')]:
    T=Q*mp.diag([1-eps,mp.mpf('0.3')])*Q.T
    # W: nearly singular
    for delta in [mp.mpf(1),mp.mpf('1e-4')]:
        W=mp.matrix([[1,1],[0,delta]])  # columns v1=(1,0), v2=(1,delta)
        A=W*T*W.T; B=W*(mp.eye(2)-T)*W.T
        # closed form: 'h' functional calculus on T
        E,Qt=mp.eigsy(T)
        h=lambda t: t*mp.log(t/(1-t)) if t>0 else mp.mpf(0)
        H=Qt*mp.diag([h(E[i]) for i in range(2)])*Qt.T
        pred=sum((W*H*W.T)[i,i] for i in range(2))   # dX=1: Tr_Y = full trace
        vals=[Dreg(A,B,flog,mp.mpf(10)**(-k)) for k in (30,40,50)]
        print('eps',mp.nstr(eps,3),'delta',mp.nstr(delta,3),'pred',mp.nstr(pred,15),'reg',[mp.nstr(v,15) for v in vals])
