import numpy as np, warnings
from schur import schur_norm
from lib import Mdd
warnings.filterwarnings("ignore")
L=np.log
h=lambda t:t*L(t/(1-t)); hp=lambda t:L(t/(1-t))+1/(1-t)
al=1.5
hr=lambda t:t**al*(1-t)**(1-al); hrp=lambda t:t**(al-1)*(1-t)**(-al)*(al-t)
for name,(g,gp) in {'BS':(h,hp),'Renyi1.5':(hr,hrp)}.items():
  for eps in [0.4,0.2,0.1,0.02]:
    lo,hi=eps,1-eps
    M2=Mdd(g,gp,np.array([lo,hi]))
    ends=schur_norm(M2)
    grid=max(schur_norm(Mdd(g,gp,np.linspace(lo,hi,n))) for n in (5,9,15))
    print("%s eps=%.2f  h'(1-eps)=%.6f |h'(eps)|=%.6f  endpoints Schur=%.6f  gridsup=%.6f"%(name,eps,gp(hi),abs(gp(lo)),ends,grid))
