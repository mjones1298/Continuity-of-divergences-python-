import numpy as np
d=2
def hp(M,a,fl=1e-30):
    w,V=np.linalg.eigh((M+M.conj().T)/2); w=np.clip(w,fl,None); return (V*w**a)@V.conj().T
def ptrB(X): return np.einsum('ajbj->ab',X.reshape(d,d,d,d))
def choi(K):
    J=np.zeros((d*d,d*d),complex)
    for k in K:
        v=sum(np.kron(np.eye(d)[i],k@np.eye(d)[i]) for i in range(d)); J+=np.outer(v,v.conj())
    return J
def dep(J,p): return (1-p)*J+p*np.eye(4)/2
def geo(J1,J2):
    A=J1+1e-14*np.eye(4);B=J2+1e-14*np.eye(4)
    Ah=hp(A,.5);Aih=hp(A,-.5); G=Ah@hp(Aih@B@Aih,.5)@Ah
    return np.min(np.linalg.eigvalsh(ptrB(G)))
def uhlF(A,B):
    sA=hp(A,.5); return np.sum(np.sqrt(np.clip(np.linalg.eigvalsh(sA@B@sA),0,None)))**2
