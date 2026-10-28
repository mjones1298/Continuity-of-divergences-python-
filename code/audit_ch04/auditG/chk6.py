import numpy as np
# (M4) Kraus rank of generalised amplitude damping (Khatri-Sharma-Wilde 2020 Kraus operators A1..A4)
def gad_kraus(g,N):
    s=np.sqrt
    A1=s(1-N)*np.diag([1,s(1-g)]); A2=s(g*(1-N))*np.array([[0,1],[0,0]])
    A3=s(N)*np.diag([s(1-g),1]); A4=s(g*N)*np.array([[0,0],[1,0]])
    return [A1,A2,A3,A4]
def choi_rank(K):
    J=sum(np.outer(k.reshape(-1),k.reshape(-1).conj()) for k in K)
    return np.linalg.matrix_rank(J,tol=1e-12), np.linalg.eigvalsh(J).min()
for g,N in [(0.3,0.2),(1.0,0.2),(0.3,1.0),(0.3,0.0),(0.0,0.3),(1.0,1.0),(0.5,0.5)]:
    print(f"p={g},N={N}: Choi rank={choi_rank(gad_kraus(g,N))[0]}")
# (M16) Cor 12.6: V=(e1+e2)/sqrt2 (dim X=1, dim Y=2), H=diag(0,2*pi): U_n V in C V for n=1
from scipy.linalg import expm
V=np.array([1,1])/np.sqrt(2); H=np.diag([0,2*np.pi])
for n in [1,2,3,10]:
    u=expm(1j*H/n)@V; print("n=",n,"U_nV parallel to V:",abs(abs(np.vdot(V,u))-1)<1e-12)
# (M3) affine f(t)=c(t-1): f(0+)+f'(inf)=0 -> jump zero; non-affine example f(t)=(t-1)^2/(1+t)?? (bounded) jump=f(0+)+f'(inf)
f=lambda t:(t-1)**2/(1+t)
print("f(0+)=",f(1e-12),"f'(inf)~",f(1e9)/1e9," sum=",f(1e-12)+f(1e9)/1e9)
