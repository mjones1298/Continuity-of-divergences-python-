import numpy as np; from lib import *
def hlog(M): w,V=np.linalg.eigh(M); return (V*np.log(np.clip(w,1e-300,None)))@V.conj().T
def rx(t): return np.array([[np.cos(t/2),-1j*np.sin(t/2)],[-1j*np.sin(t/2),np.cos(t/2)]])
def um(A,B): return np.real(np.trace(A@(hlog(A)-hlog(B))))
def bs(A,B):
    Ah=hp(A,.5); X=Ah@hp(B,-1)@Ah; return np.real(np.trace(Ah@hlog(X)@Ah))
def petz(A,B,a): return np.log(np.real(np.trace(hp(A,a)@hp(B,1-a))))/(a-1)
def geoS(A,B): # -2 log Tr(A#B) for states
    Ah=hp(A,.5);Aih=hp(A,-.5); return -2*np.log(np.real(np.trace(Ah@hp(Aih@B@Aih,.5)@Ah)))
dl=1e-2
print("Choi STATES of Rx(pi/2)+dep_p vs Rx(pi/2+0.01)+dep_p; divergence / (delta^2/4)")
print("p        geo(-2logTr#)   BS        Umegaki    Petz a=1/2   -logUhlF")
for p in [1e-2,1e-3,1e-4,1e-5,1e-6]:
    A=dep(choi([rx(np.pi/2)]),p)/2; B=dep(choi([rx(np.pi/2+dl)]),p)/2
    s=dl**2/4
    print(f"{p:.0e}  {geoS(A,B)/s:10.2f}  {bs(A,B)/s:10.2f}  {um(A,B)/s:8.2f}  {petz(A,B,.5)/s:8.3f}  {-np.log(uhlF(A,B))/s:8.3f}")
