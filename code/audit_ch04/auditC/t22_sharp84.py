"""Sharpness of Thm 8.4 constants on a structured instance: V1=|0><0|, V2=|1><1| (dX=dY=2), nu^2=1/2.
Upper bound |||Psi_K|||_1 <= ||Omega*(1)|| ||K|| attained at K=1_r; lower bound nu^2 ||K|| <= |||Psi_K|||_1 tested at K=sigma_x."""
import numpy as np
from diamond import *
V1=np.array([[1,0],[0,0]],dtype=complex); V2=np.array([[0,0],[0,1]],dtype=complex)
Ks=[V1,V2]; W=W_of(Ks)
print('nu_est^2 =',nu_est(Ks,restarts=30)**2)
for name,K in (('1',np.eye(2,dtype=complex)),('sigma_x',np.array([[0,1],[1,0]],dtype=complex)),('sigma_z',np.diag([1,-1]).astype(complex))):
    d=diamond_herm(W@K@W.conj().T,2,2)
    print(name,'diamond =',d,' nu^2||K|| = 0.5,  ||Om*(1)|| ||K|| = 1')
