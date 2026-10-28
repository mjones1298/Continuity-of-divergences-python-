# Exact incoherent family at qubit depolarized gates: J1e = J1 + e(aa* - xx*), a=vec(R)/sqrt2, x=vec(XR)/sqrt2.
# On span{a,x}: A_e=diag(lam+e, mu-e), B = mu + (lam-mu) bb*, b = c a + i s x;  F = mu + tr(A_e # B)/2.
import mpmath as mp
mp.mp.dps=40
def F(p,x,e):
    lam=2-1.5*p; mu=p/2
    s2=x*p*(4-3*p)/(4*(1-p)**2); s=mp.sqrt(s2); c=mp.sqrt(1-s2)
    A=mp.matrix([[lam+e,0],[0,mu-e]])
    bb=mp.matrix([[c*c, -1j*c*s],[1j*c*s, s*s]])   # b b^* with b=(c, i s)
    B=mu*mp.eye(2)+(lam-mu)*bb
    al=mp.sqrt(mp.det(A)); be=mp.sqrt(mp.det(B).real)
    Mx=A/al+B/be
    G=mp.sqrt(al*be)*Mx/mp.sqrt(mp.det(Mx).real)
    return mu+ (G[0,0]+G[1,1]).real/2
def rate(p,x):
    h=mp.mpf(p)*mp.mpf('1e-12'); return -(F(p,x,h)-F(p,x,-h))/(2*h)   # d = e exactly
# check against closed form at e=0
p=mp.mpf('1e-3'); x=mp.mpf(1)
print("F(e=0) =",F(p,x,0)," closed:",p/2+(1-p/2)/mp.sqrt(1+x))
for p in ['1e-2','1e-4','1e-6']:
    p=mp.mpf(p)
    row=[(float(x), float(rate(p,x)*p)) for x in [mp.mpf('0.25'),1,2,4,mp.mpf('8')]]
    print("p=",mp.nstr(p,2)," rate*p:",[ (a,round(b,6)) for a,b in row])
# guess: rate*p -> sqrt(x)/(2 (1+x)^{3/2}) * something ; print ratio
p=mp.mpf('1e-8')
for x in [mp.mpf('0.25'),1,2,4,8,mp.mpf(1)/2]:
    r=rate(p,x)*p
    print(float(x), float(r), float(r/(mp.sqrt(x)/(1+x)**1.5)), float(r/(x/(1+x)**1.5)),float(r/(mp.sqrt(x)/(1+x))))
# -2 log F rate: 2*rate/F, maximise over x
best=max((float(2*rate(p,x)*p/F(p,x,0)),float(x)) for x in [mp.mpf(k)/4 for k in range(1,200)])
print("max over x of p * d(-2logF)/dd :",best)
