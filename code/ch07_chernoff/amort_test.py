import numpy as np, scipy.linalg as sla
from scipy.optimize import minimize
rng = np.random.default_rng(7)
def kraus(g): return [np.array([[1,0],[0,np.sqrt(1-g)]]), np.array([[0,np.sqrt(g)],[0,0]])]
def rootF(A, B):
    sA = sla.sqrtm(A); return np.real(np.trace(sla.sqrtm(sA@B@sA)))
def rootF2(A,B):
    w,U=np.linalg.eigh(A); sA=(U*np.sqrt(np.clip(w,0,None)))@U.conj().T
    s=np.linalg.svd(sA@sla.sqrtm(B) if False else None,compute_uv=False) if False else None
    M=sA@B@sA; w2=np.linalg.eigvalsh((M+M.conj().T)/2); return np.sum(np.sqrt(np.clip(w2,0,None)))
def apply(Ks, rho, dR):
    return sum(np.kron(K, np.eye(dR))@rho@np.kron(K, np.eye(dR)).conj().T for K in Ks)
def state(x, d):
    A = (x[:d*d]+1j*x[d*d:]).reshape(d, d); r = A@A.conj().T; return r/np.trace(r)
def ratio(x, g, h, dR):
    d = 2*dR; rho = state(x[:2*d*d], d); sig = state(x[2*d*d:], d)
    Fin = rootF2(rho, sig)
    if Fin < 1e-6: return 10.0
    return rootF2(apply(kraus(g), rho, dR), apply(kraus(h), sig, dR))/Fin
if __name__ == "__main__":
    for (g, h) in [(0.5, 0.4), (0.5, 0.2), (0.5, 0.05), (0.9, 0.3), (0.2, 0.15), (0.05, 0.6)]:
      f = np.sqrt(g*h)+np.sqrt((1-g)*(1-h))
      for dR in [1, 2, 3]:
          d = 2*dR; best = 10
          for trial in range(30):
              x0 = rng.normal(size=4*d*d)
              r = minimize(ratio, x0, args=(g, h, dR), method='L-BFGS-B', options={'maxiter': 3000})
              best = min(best, r.fun)
          print("g=%.2f h=%.2f dR=%d   min ratio sqrtF_out/sqrtF_in = %.10f   f = %.10f   diff=%.2e" % (g, h, dR, best, f, best-f), flush=True)
