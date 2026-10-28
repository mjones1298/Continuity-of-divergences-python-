import numpy as np
from diamond import *
# sanity: replacement channels difference
dX,dY=2,2
th=0.3
y1=np.array([1,0],complex); y2=np.array([np.cos(th),np.sin(th)],complex)
phi=np.outer(y1,y1.conj()); psi=np.outer(y2,y2.conj())
J=np.kron(phi-psi,np.eye(dX))
print('diamond(Phi1-Psi1)=',diamond_herm(J,dY,dX),' expect',2*np.sin(th))
