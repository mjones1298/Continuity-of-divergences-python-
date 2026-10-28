import mpmath as mp
import numpy as np
mp.mp.dps = 60

def tomp(A):
    A = np.asarray(A)
    M = mp.matrix(A.shape[0], A.shape[1])
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            M[i, j] = mp.mpc(float(A[i, j].real), float(A[i, j].imag)) if np.iscomplexobj(A) else mp.mpf(float(A[i, j]))
    return M

def mp_herm(M): return (M + M.H) / 2

def mp_fun(M, f):
    E, Q = mp.eigh(mp_herm(M))
    D = mp.diag([f(E[i]) for i in range(len(E))])
    return Q * D * Q.H

def mp_reg_mean(A, B, fx, eta):
    """(A+eta) sigma (B+eta) with representing function fx (mp)"""
    n = A.rows
    I = mp.eye(n)
    Ae = A + eta * I; Be = B + eta * I
    Ah = mp_fun(Ae, mp.sqrt); Aih = mp_fun(Ae, lambda x: 1 / mp.sqrt(x))
    M = Aih * Be * Aih
    return mp_herm(Ah * mp_fun(M, fx) * Ah)

def to_np(M):
    return np.array([[complex(M[i, j]) for j in range(M.cols)] for i in range(M.rows)])

# connections (mp versions of f)
def f_geo(x): return mp.sqrt(x)
def f_harm(x): return 2 * x / (1 + x)
def f_log(x): return (x - 1) / mp.log(x) if abs(x - 1) > mp.mpf(10) ** (-40) else mp.mpf(1)
def f_a(al): return lambda x: x ** al
def f_mix(x): return mp.mpf('0.3') + mp.mpf('0.2') * x + mp.mpf('0.5') * 2 * x / (1 + x)
def f_skew(x): return mp.mpf('0.4') + mp.mpf('0.6') * mp.sqrt(x)
def f_arith(x): return (1 + x) / 2
