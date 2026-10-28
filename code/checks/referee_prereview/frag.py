import numpy as np
from scipy.linalg import sqrtm, logm, fractional_matrix_power as fmp
np.set_printoptions(precision=4)
def psd_pow(A,a):
    w,V=np.linalg.eigh((A+A.conj().T)/2); w=np.clip(w,0,None)
    with np.errstate(divide='ignore'):
        wp=np.where(w>1e-300,w**a,0.0)
    return (V*wp)@V.conj().T
def gmean(A,B):
    # A#B for A>0
    Ah=psd_pow(A,.5); Aih=psd_pow(A,-.5)
    return Ah@psd_pow(Aih@B@Aih,.5)@Ah
def geo_fid(A,B):  # Tr(A#B), root geometric fidelity; regularize if singular
    return np.real(np.trace(gmean(A,B)))
def uhl(A,B):
    sA=psd_pow(A,.5); w=np.linalg.eigvalsh(sA@B@sA); return np.sum(np.sqrt(np.clip(w,0,None)))
def tn(A): return np.sum(np.abs(np.linalg.eigvalsh(A)))
def BS(A,B):
    Ah=psd_pow(A,.5); Bih=psd_pow(B,-.5)
    X=Ah@Bih@Bih@Ah  # A^{1/2} B^{-1} A^{1/2}
    w,V=np.linalg.eigh(X); 
    return np.real(np.trace(A@ (V*np.log(np.clip(w,1e-300,None)))@V.conj().T))
def umeg(A,B):
    def lg(M):
        w,V=np.linalg.eigh(M); return (V*np.log(w))@V.conj().T
    return np.real(np.trace(A@(lg(A)-lg(B))))
def hs(A,B,g): # hockey stick E_gamma = Tr(A-gB)_+
    w=np.linalg.eigvalsh(A-g*B); return np.sum(w[w>0])
def tau(p,th):
    v=np.array([np.cos(th),np.sin(th)]); vp=np.array([-np.sin(th),np.cos(th)])
    return (1-p)*np.outer(v,v)+p*np.outer(vp,vp)
print("Reset-channel pair: thermal residual p, rotation theta. diamond dist = ||tau-tau'||_1")
print("p, theta, diamond, 1-Tr(#) [geo], 1-sqrtF Uhlmann, D_BS, D_Umegaki, E_gamma=1.05")
for p in [1e-2,1e-3,1e-4,1e-6]:
    for th in [1e-3,3e-3,1e-2,3e-2,1e-1]:
        A=tau(p,0);B=tau(p,th)
        print(f"{p:.0e} {th:.0e} dia={tn(A-B):.2e} geo={1-geo_fid(A,B):.3e} uhl={1-uhl(A,B):.3e} BS={BS(A,B):.3e} Um={umeg(A,B):.3e} hs={hs(A,B,1.05):.3e}")
# local Lipschitz-type ratio: (1-geo)/diamond^2 vs p ; theory RLD ~ 1/p
print("\nratio (1-geo)/theta^2 at theta=1e-4:")
for p in [1e-1,1e-2,1e-3,1e-4]:
    A=tau(p,0);B=tau(p,1e-4); print(p,(1-geo_fid(A,B))/1e-8, (1-uhl(A,B))/1e-8)
# crossover theta* where 1-geo reaches 0.5
print("\ncrossover theta* where 1-Tr(#)=0.1 :")
for p in [1e-2,1e-3,1e-4,1e-5]:
    ths=np.logspace(-6,0,2000)
    v=[1-geo_fid(tau(p,0),tau(p,t)) for t in ths]
    i=np.argmax(np.array(v)>0.1); print(p, ths[i], np.sqrt(p))
