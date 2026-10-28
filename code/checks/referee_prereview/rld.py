import numpy as np; from lib import *
def rx(t): return np.array([[np.cos(t/2),-1j*np.sin(t/2)],[-1j*np.sin(t/2),np.cos(t/2)]])
h=1e-6
print("RLD Fisher value of channel (KW 2008.11178): ||Tr_Y[dJ J^{-1} dJ]||_inf  for Rx(pi/2+theta)+dep p, theta=0")
for p in [1e-1,1e-2,1e-3,1e-4]:
    J=dep(choi([rx(np.pi/2)]),p); dJ=(dep(choi([rx(np.pi/2+h)]),p)-dep(choi([rx(np.pi/2-h)]),p))/(2*h)
    M=ptrB(dJ@np.linalg.inv(J)@dJ); F=np.max(np.linalg.eigvalsh((M+M.conj().T)/2))
    print(f"p={p:.0e}: F_RLD={F:.3e}  F_RLD*p={F*p:.3f}")
