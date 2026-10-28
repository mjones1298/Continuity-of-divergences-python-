import numpy as np; from lib import *
def rx(t): return np.array([[np.cos(t/2),-1j*np.sin(t/2)],[-1j*np.sin(t/2),np.cos(t/2)]])
def hlog(M): w,V=np.linalg.eigh((M+M.conj().T)/2); return (V*np.log(np.clip(w,1e-300,None)))@V.conj().T
def bs(A,B): Ah=hp(A,.5); return np.real(np.trace(Ah@hlog(Ah@hp(B,-1)@Ah)@Ah))
def um(A,B): return np.real(np.trace(A@(hlog(A)-hlog(B))))
def h(x): return -x*np.log(x)-(1-x)*np.log(1-x)
print("Stein: BS channel converse (Choi-input lower bound on it) vs paper-3 upper bound on true adaptive zeta; nats")
for p in [1e-3,1e-4]:
    J0=dep(choi([rx(np.pi/2)]),p); kap=np.max(np.linalg.eigvalsh(J0))/np.min(np.linalg.eigvalsh(J0))
    for dl in [3e-3,1e-2,3e-2]:
        J1=dep(choi([rx(np.pi/2+dl)]),p); e=np.sin(dl/2)  # half diamond distance of the two unitaries (depol contracts it)
        ub=2*e*np.log(2)+(1+e)*h(e/(1+e))+e*np.log(kap)
        print(f"p={p:.0e} delta={dl:.0e}: D_BS(Choi)>={bs(J1/2,J0/2):.3f}  D_Um(Choi)={um(J1/2,J0/2):.4f}  paper3 bound zeta<={ub:.3f}  ratio BS/zeta_ub>={bs(J1/2,J0/2)/ub:.1f}")
