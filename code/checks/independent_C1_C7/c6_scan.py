from pathlib import Path
import numpy as np, mpmath as mp
from qlib import *
src=open(Path(__file__).resolve().parent / 'c6.py').read().split('results=[]')[0]; ns={}; exec(src,ns); bt=ns['bound_terms']
mp.mp.dps=80
def mpP(v): return mpouter(v)
mx=(0,None); n=0
for th in ['0.001','0.1','0.7','1.2','1.5707963']:
  t_=mp.mpf(th); J0=mpkron(mpP([1,0]),mp.eye(2)); Jt=mpkron(mpP([mp.cos(t_),mp.sin(t_)]),mp.eye(2))
  for s in 10**np.linspace(-8,-0.01,25):
    for t in sorted(set([s,s/10,min(10*s,0.9),1-s])):
      S=mp.mpf(s);T=mp.mpf(t)
      A2=(1-S)*J0+S*Jt; B2=(1-T)*Jt+T*J0
      F2=fid_mp(A2,B2,2,2,'0',dps=80)   # full-rank exact
      F1=0  # disjoint supports; verified in C1
      tau,tt,nrm,_,_=bt(mp2np(J0),mp2np(Jt),mp2np(A2),mp2np(B2))
      rhs=nrm*(np.sqrt(tau)+np.sqrt(2)*tt); lhs=float(F2); n+=1
      if lhs/rhs>mx[0]: mx=(lhs/rhs,(th,s,t,lhs,rhs,tau))
print('scan n=',n,'max ratio',mx)
print('analytic check s=t=1e-4: F2=2sqrt(s(1-s))?')
J0=mpkron(mpP([1,0]),mp.eye(2)); t_=mp.mpf('0.7'); Jt=mpkron(mpP([mp.cos(t_),mp.sin(t_)]),mp.eye(2)); S=mp.mpf('1e-4')
print(fid_mp((1-S)*J0+S*Jt,(1-S)*Jt+S*J0,2,2,'0',80), 2*mp.sqrt(S*(1-S)))
