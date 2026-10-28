import numpy as np, mpmath as mp
from qlib import *
mp.mp.dps=60
worst=0
for p in ['0.5','0.1','0.01','0.001','0.9']:
  for th in ['0.01','0.3','1.0','1.5707963267948966','2.5']:
    P_=mp.mpf(p); t=mp.mpf(th)
    def sig(v): return (1-P_)*mpouter(v)+P_/2*mp.eye(2)
    J0=mpkron(sig([1,0]),mp.eye(2)); Jt=mpkron(sig([mp.cos(t),mp.sin(t)]),mp.eye(2))
    f=fid_mp(J0,Jt,2,2,'0',dps=60)
    pred=(1+(1-P_)**2*mp.sin(t)**2/(2*P_-P_**2))**(-mp.mpf(1)/2)
    worst=max(worst,abs(f-pred))
    if th in ('0.3',): 
        s=fid_sdp(mp2np(J0),mp2np(Jt),2,2)
        print(p,th,mp.nstr(f,15),mp.nstr(pred,15),'SDP',s)
print('max |F-pred| over grid', mp.nstr(worst,5))
