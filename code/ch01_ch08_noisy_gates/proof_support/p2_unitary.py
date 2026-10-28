# P2 (Prop. L2(b)): exact supremum of the absolute slope over unitary Phi, and an SDP check of the
# optimal direction at the unitary optimum. This is the first part of the original p2_worst.py, unchanged;
# its double-precision search over mixed Phi is not included (see code/EXCLUDED.md; use p2_search_v2.py).
import sys
from pathlib import Path
exec(open(Path(__file__).resolve().parent / 'p2_common.py').read())
def unitary_exact(p,d=2):
    lam=p/2; b1=2-1.5*p
    th=np.linspace(1e-6,np.pi/2,200001); c=np.cos(th)**2; s=np.sin(th)**2
    val=np.sqrt(d)/4*(c/b1**2+s/lam**2)/(c/b1+s/lam)**1.5
    return val.max(), th[val.argmax()]
for p in [1e-2,1e-3,1e-4]:
    v,th=unitary_exact(p); print(f'p={p}: unitary-Phi exact sup = {v:.6f} -> p*sup = {p*v:.6f}  (limit 2/(3sqrt3)={2/(3*np.sqrt(3)):.6f}), theta*={th:.4e}')
# verify formula with SDP direction at the optimal unitary
p=1e-2; v,th=unitary_exact(p); V=np.cos(th)*np.eye(2)-1j*np.sin(th)*X
B=JDp(p); G=grad_op(choi([V]),B,1e-8); print(' SDP check at unitary optimum: p*sup_E =',p*best_dir(G))
