import numpy as np; from lib import *
def rx(t): return np.array([[np.cos(t/2),-1j*np.sin(t/2)],[-1j*np.sin(t/2),np.cos(t/2)]])
def td(J1,J2): return np.sum(np.abs(np.linalg.eigvalsh((J1-J2)/2)))/2
print("Rx(pi/2)+depol p vs Rx(pi/2+delta)+depol p")
for p in [1e-2,1e-3,1e-4]:
    for dl in [1e-3,1e-2,3e-2]:
        J1=dep(choi([rx(np.pi/2)]),p); J2=dep(choi([rx(np.pi/2+dl)]),p)
        print(f"p={p:.0e} delta={dl:.0e}: -2log F_geo={-2*np.log(geo(J1,J2)):.4f}  -log F_Uhl(Choi)={-np.log(uhlF(J1/2,J2/2)):.2e}  ChoiTD={td(J1,J2):.2e}")
