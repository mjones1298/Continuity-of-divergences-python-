# Checks: (1) min of t log(t/(1-t)); (2) Loewner data of Table 1 via numerical integration; (3) Example one-parameter sharp
import numpy as np, mpmath as mp
from scipy.integrate import quad
from scipy.optimize import minimize_scalar

# (1)
r = minimize_scalar(lambda t: t*np.log(t/(1-t)), bounds=(1e-9,1-1e-9), method='bounded', options={'xatol':1e-14})
print("min h_BS:", r.x, r.fun)

# (2) Loewner representation f(x) = a + b(x-1) + c(x-1)^2 + int (x-1)^2/(x+s) dmu(s)
def rep(a,b,c,dens,x, split=()):
    pts=[0]+list(split)+[np.inf]
    tot=0
    for lo,hi in zip(pts[:-1],pts[1:]):
        v,_=quad(lambda s:(x-1)**2/(x+s)*dens(s),lo,hi,limit=400)
        tot+=v
    return a+b*(x-1)+c*(x-1)**2+tot
tests = {
 'BS x log x': (lambda x:x*np.log(x), 0,1,0, lambda s:s/(1+s)**2),
 'revBS -log x': (lambda x:-np.log(x), 0,-1,0, lambda s:1/(1+s)**2),
 'Jeffreys (x-1)log x': (lambda x:(x-1)*np.log(x), 0,0,0, lambda s:1/(1+s)),
 'Hellinger (sqrt x -1)^2': (lambda x:(np.sqrt(x)-1)**2, 0,0,0, lambda s:2/np.pi*np.sqrt(s)/(1+s)**2),
 'JS': (lambda x:x*np.log(x)-(1+x)*np.log((1+x)/2), 0,0,0, lambda s:min(s,1)/(1+s)**2),
 'Renyi a=1.5': (lambda x:x**1.5, 1,1.5,0, lambda s:np.sin(np.pi*0.5)/np.pi*s**1.5/(1+s)**2),
 'Renyi a=1.3': (lambda x:x**1.3, 1,1.3,0, lambda s:np.sin(np.pi*0.3)/np.pi*s**1.3/(1+s)**2),
 'Renyi a=-0.5': (lambda x:x**-0.5, 1,-0.5,0, lambda s:(-np.sin(np.pi*(-0.5)))/np.pi*s**-0.5/(1+s)**2),
 'Renyi a=-0.3': (lambda x:x**-0.3, 1,-0.3,0, lambda s:(-np.sin(np.pi*(-0.3)))/np.pi*s**-0.3/(1+s)**2),
}
for name,(f,a,b,c,dens) in tests.items():
    errs=[]
    for x in [0.3,0.9,2.0,5.0]:
        errs.append(rep(a,b,c,dens,x,split=(1,))-f(x))
    print(name, ["%.2e"%e for e in errs])
