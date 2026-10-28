import numpy as np
from schur import schur_norm
from scipy.optimize import brentq

def kappa(s):
    if np.isinf(s):
        k = lambda t: (2*t-1)**2/(1-t)
        kp = lambda t: (2*t-1)*(3-2*t)/(1-t)**2
    else:
        k = lambda t: (2*t-1)**2/(s*(1-t)+t)
        def kp(t):
            w = s*(1-t)+t
            return (2*t-1)*(2/w+(1+s)/w**2)
    return k,kp

def Mdd(g,gp,Lam):
    return np.array([[ (g(x)-g(y))/(x-y) if abs(x-y)>1e-12 else gp(x) for y in Lam] for x in Lam])

def schur2_formula(q1,q,q2):
    L = max(abs(q1),abs(q2))
    A = lambda t: np.sqrt(max((t-q1)*(t-q2),0)); B = lambda t: np.sqrt(max((t+q1)*(t+q2),0))
    f = lambda t: A(t)+B(t)-2*abs(q)
    if f(L) >= 0: return L
    hi = L+1
    while f(hi) < 0: hi *= 2
    return brentq(f, L, hi, xtol=1e-14)
