import numpy as np, sys
sys.path.insert(0,'.')
from lib import *
from mplib import *
from mpdil import mp, build  # sets dps
import mpmath as mpm
mp.mp.dps = 100

I2=np.eye(2,dtype=complex); X=np.array([[0,1],[1,0]],complex); Y=np.array([[0,-1j],[1j,0]]); Z=np.diag([1,-1]).astype(complex)

def choi_np(kraus, dX=2):
    return sum(np.outer(vec(K), vec(K).conj()) for K in kraus)

def Fhat_exact(JP, JQ, dY=2, dX=2):
    """lambda_min Tr_Y (JP # JQ) computed via S-frame, double precision (fine if S well conditioned)."""
    M = ka_frame_conn(JP, JQ, make_conns()['geo'], tol=1e-9)
    return np.linalg.eigvalsh(herm(ptrace_Y(M, dY, dX)))[0]

# ---------- Test A: unitary qubit channel, U_n -> 1 ----------
V = np.array([[1,1],[1,-1]],complex)/np.sqrt(2)  # Hadamard
JV = choi_np([V])
print("Unitary channel: F(Phi,Phi)=", Fhat_exact(JV, JV))
for n in [1,10,100,1000]:
    H = Z
    Un = np.array([[np.exp(1j/n),0],[0,np.exp(-1j/n)]]) 
    JU = choi_np([Un @ V])
    print("  n=%5d  |J(Phi_n)-J(Phi)|_1=%.2e  F(Phi,Phi_n)=%.3e" % (n, np.linalg.norm(JU-JV, 'nuc'), Fhat_exact(JV, JU)))
