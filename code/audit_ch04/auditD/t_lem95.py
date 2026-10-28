import numpy as np, warnings
from schur import schur_norm
from scipy.optimize import brentq
warnings.filterwarnings("ignore")
rng = np.random.default_rng(1)

def schur2_formula(q1,q,q2):
    L = max(abs(q1),abs(q2))
    A = lambda t: np.sqrt(max((t-q1)*(t-q2),0)); B = lambda t: np.sqrt(max((t+q1)*(t+q2),0))
    f = lambda t: A(t)+B(t)-2*abs(q)
    if f(L) >= 0: return L
    hi = L+1
    while f(hi) < 0: hi *= 2
    return brentq(f, L, hi, xtol=1e-14)


if __name__=="__main__":
    # (c) random triples
    maxerr=0
    for _ in range(60):
        q1,q,q2 = rng.normal(size=3)*rng.choice([0.1,1,5])
        s = schur_norm([[q1,q],[q,q2]])
        f = schur2_formula(q1,q,q2)
        maxerr = max(maxerr, abs(s-f)/max(1,abs(s)))
    print("(c) max rel err formula vs SDP:", maxerr)
    
    # (d) q1<=q<=q2: ratio ||M||_S / max(|q1|,|q2|) <= sqrt2 ?
    worst=0; arg=None
    for _ in range(400):
        a = np.sort(rng.uniform(-1,1,size=3))
        q1,q,q2 = a
        L = max(abs(q1),abs(q2))
        if L<1e-6: continue
        r = schur2_formula(q1,q,q2)/L
        if r>worst: worst=r; arg=(q1,q,q2)
    print("(d) worst ratio (formula), should be <= sqrt2=%.6f:"%np.sqrt(2), worst, arg)
    # extremal: (-1,1,1) ; also try boundary grid
    best=0
    for q1 in np.linspace(-1,1,81):
        for q2 in np.linspace(q1,1,81):
            for q in np.linspace(q1,q2,41):
                L=max(abs(q1),abs(q2))
                if L<1e-9: continue
                best=max(best, schur2_formula(q1,q,q2)/L)
    print("grid sup ratio:", best)
    # SDP check of the extremal example
    print(schur_norm([[-1,1],[1,1]]))
    # what if q1<=q<=q2 fails (q outside): ratio can exceed sqrt2?
    w=0
    for _ in range(2000):
        q1,q,q2 = rng.uniform(-1,1,3)
        L=max(abs(q1),abs(q2)); 
        w=max(w, schur2_formula(q1,q,q2)/L)
    print("without ordering hypothesis, worst ratio (random):", w)
