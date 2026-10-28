# Symbolic: d/de [tr(A_e # B)/2] at e=0, leading order in p, with s^2 = x p (4-3p)/(4(1-p)^2)
import sympy as sp
p,x,e=sp.symbols('p x e',positive=True)
lam=2-sp.Rational(3,2)*p; mu=p/2
s2=x*p*(4-3*p)/(4*(1-p)**2); c2=1-s2
# work in real basis (conjugate by diag(1,i)): B=[[mu+(lam-mu)c2, (lam-mu)cs],[.., mu+(lam-mu)s2]]
cs=sp.sqrt(c2*s2)
A=sp.Matrix([[lam+e,0],[0,mu-e]]); B=sp.Matrix([[mu+(lam-mu)*c2,(lam-mu)*cs],[(lam-mu)*cs,mu+(lam-mu)*s2]])
al=sp.sqrt(A.det()); be=sp.sqrt(sp.simplify(B.det()))
M=A/al+B/be
trG=sp.sqrt(al*be)*M.trace()/sp.sqrt(M.det())
F=mu+trG/2
dF=sp.diff(F,e).subs(e,0)
lead=sp.limit(sp.simplify(dF*p),p,0)
print("lim_{p->0} p dF/de =",sp.simplify(lead))
F0=sp.simplify(F.subs(e,0)); print("F(e=0) =",sp.simplify(F0), " check closed form diff:",sp.simplify(F0-(p/2+(1-p/2)/sp.sqrt(1+x))))
