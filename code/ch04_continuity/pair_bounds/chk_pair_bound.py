"""Check (a): A#B = C^{1/2} g(Ttilde) C^{1/2} incl. singular C; (b) candidate bound;
compare with v5 Thm 10.4(b) and with the improved sandwich constant (Lemma S below)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))  # shared modules in code/common (glib.py)
import numpy as np, glib as g
rng = np.random.default_rng(7)
gs = lambda t: np.sqrt(np.clip(t,0,1)*np.clip(1-t,0,1))
def rel_deriv(A, C, tol=1e-10):
    a = g.sqrtm(C); ap = g.fn(C, lambda w: np.where(w>tol, w, np.inf)**-0.5)
    return a, g.herm(ap@A@ap)
worst_a = 0; worst_b = []; worst_lemma = 0
for trial in range(400):
    dX, dY = rng.integers(1,3), rng.integers(2,4)
    ranks = rng.integers(1, 4, size=4)
    Ks = [g.rand_kraus(rng, dY, dX, r) for r in ranks]
    sc = rng.uniform(0.2, 1.5, size=4)
    Js = [s*g.choi(K) for s, K in zip(sc, Ks)]
    if trial % 3 == 0:   # make second pair close to first pair (incl. different sums)
        e = 10**rng.uniform(-6, -1)
        Js[2] = Js[0] + e*g.choi(g.rand_kraus(rng, dY, dX, 2)); Js[3] = Js[1]*(1+e*rng.uniform())
    JO = sum(Js); W = g.frame(JO); r = W.shape[1]
    A1, B1, A2, B2 = [g.arn(W, J) for J in Js]
    C1, C2 = A1+B1, A2+B2
    out = []
    for A, B, J1, J2 in [(A1, B1, Js[0], Js[1]), (A2, B2, Js[2], Js[3])]:
        C = A+B; a, Tt = rel_deriv(A, C)
        lhs = g.gmean(A, B); rhs = a@g.fn(Tt, gs)@a
        worst_a = max(worst_a, np.abs(lhs-rhs).max())
        # also Choi level: J1#J2 = W a g(T) a W^*
        worst_a = max(worst_a, np.abs(g.gmean(J1, J2) - W@rhs@W.conj().T).max())
        out.append((a, Tt, C))
    (a1, T1, _), (a2, T2, _) = out
    F1 = g.Fhat(Js[0], Js[1], dY, dX); F2 = g.Fhat(Js[2], Js[3], dY, dX)
    Om = np.linalg.eigvalsh(g.trY(JO, dY, dX)).max()      # ||Omega'^*(1)||  (sum of four)
    tau = np.abs(np.linalg.eigvalsh(T1-T2)).max(); dC = np.abs(np.linalg.eigvalsh(C1-C2)).max()
    cand = Om*(max(np.linalg.eigvalsh(C1).max(), np.linalg.eigvalsh(C2).max())*np.sqrt(tau) + np.sqrt(dC))
    Ombar = Om/2
    v5 = Ombar*(np.sqrt(tau) + np.sqrt(2)*dC)            # v5 Thm 10.4(b), tanh(dT/2)=||C1-C2||
    s = np.sqrt(max(0, 1-dC**2)); imp = Ombar*(np.sqrt(tau) + np.sqrt(2)*dC/np.sqrt(1+s))
    worst_b.append((abs(F1-F2), cand, v5, imp))
    # Lemma S check: ||a1 phi(T1) a1 - a2 phi(T2) a2|| <= 1/2 ||phi(T1)-phi(T2)|| + sqrt2*||phi||*2gam/sqrt(1+s)
    if dC < 1-1e-9:
        for phi, nphi in [(gs, 0.5), (lambda t: t**2, 1.0), (lambda t: np.cos(3*t), 1.0)]:
            L = np.abs(np.linalg.eigvalsh(a1@g.fn(T1, phi)@a1 - a2@g.fn(T2, phi)@a2)).max()
            H = np.abs(np.linalg.eigvalsh(g.fn(T1, phi)-g.fn(T2, phi))).max()
            R = 0.5*H + np.sqrt(2)*nphi*dC/np.sqrt(1+s)
            worst_lemma = max(worst_lemma, L/R)
wb = np.array(worst_b)
print("max |A#B - a g(T~) a| :", worst_a)
print("candidate (b) violations:", np.sum(wb[:,0] > wb[:,1]+1e-9), " max ratio", (wb[:,0]/wb[:,1]).max())
print("v5 10.4(b) violations:", np.sum(wb[:,0] > wb[:,2]+1e-9), " improved violations:", np.sum(wb[:,0] > wb[:,3]+1e-9))
print("fraction where v5 bound < candidate:", np.mean(wb[:,2] <= wb[:,1]+1e-12))
print("Lemma S max LHS/RHS:", worst_lemma)
