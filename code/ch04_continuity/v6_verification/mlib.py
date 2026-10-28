# Independent helper library (mpmath, high precision). Conventions follow Watrous / Jones v5:
# J(Phi) = sum_ij Phi(E_ij) (x) E_ij on Y (x) X ; vec(E_ab) = e_a (x) e_b (row-major);
# Fhat = lambda_min Tr_Y (J1 # J2), computed EXACTLY via the ARN representation (v5 Thm 2.15):
#   J1 # J2 = W g_#(T) W^*,  W a frame of J1+J2,  T = W^+ J1 W^{+*}.
import mpmath as mp
mp.mp.dps = 60

def M(rows): return mp.matrix(rows)
def zeros(n, m=None): return mp.zeros(n, m if m is not None else n)
def eye(n): return mp.eye(n)
def dag(A): return A.transpose_conj()

def kron(A, B):
    ra, ca, rb, cb = A.rows, A.cols, B.rows, B.cols
    K = mp.zeros(ra*rb, ca*cb)
    for i in range(ra):
        for j in range(ca):
            a = A[i, j]
            if a == 0: continue
            for k in range(rb):
                for l in range(cb):
                    K[i*rb+k, j*cb+l] = a*B[k, l]
    return K

def vec(A):  # row-major
    v = mp.zeros(A.rows*A.cols, 1)
    for i in range(A.rows):
        for j in range(A.cols):
            v[i*A.cols+j] = A[i, j]
    return v

def herm(A): return (A + dag(A))/2

def eigh(A):
    E, Q = mp.eighe(herm(A))
    return [mp.re(e) for e in E], Q

def fun(A, f):
    E, Q = eigh(A)
    n = A.rows
    D = mp.zeros(n)
    for i in range(n): D[i, i] = f(E[i])
    return Q*D*dag(Q)

def choi_kraus(Ks, dX, dY):
    J = zeros(dX*dY)
    for K in Ks:
        v = vec(K); J += v*dag(v)
    return J

def ptrace_Y(J, dY, dX):
    R = zeros(dX)
    for a in range(dX):
        for b in range(dX):
            s = 0
            for y in range(dY):
                s += J[y*dX+a, y*dX+b]
            R[a, b] = s
    return R

def frame(J, tol=None):
    """W with W W^* = J, injective (columns = sqrt(eig) * eigvec for eig > tol)."""
    if tol is None: tol = mp.mpf(10)**(-mp.mp.dps//2)
    E, Q = eigh(J)
    idx = [i for i, e in enumerate(E) if e > tol*max(1, max(E))]
    W = mp.zeros(J.rows, len(idx))
    for c, i in enumerate(idx):
        for r in range(J.rows):
            W[r, c] = Q[r, i]*mp.sqrt(E[i])
    return W

def pinv_frame(W):
    return mp.inverse(dag(W)*W)*dag(W)

def gmean(A, B):
    """Kubo-Ando geometric mean of PSD A,B (possibly singular), exact via ARN."""
    W = frame(A+B)
    if W.cols == 0: return zeros(A.rows)
    Wp = pinv_frame(W)
    T = herm(Wp*A*dag(Wp))
    G = fun(T, lambda t: mp.sqrt(max(t, 0)*max(1-t, 0)))
    return W*G*dag(W)

def gmean_pd(A, B):
    Ah = fun(A, mp.sqrt); Aih = fun(A, lambda t: 1/mp.sqrt(t))
    return Ah*fun(Aih*B*Aih, lambda t: mp.sqrt(max(t, 0)))*Ah

def lmin(A): return min(eigh(A)[0])
def lmax(A): return max(eigh(A)[0])
def opnorm(A):
    E, _ = eigh(dag(A)*A); return mp.sqrt(max(E))
def tracenorm_h(A): return sum(abs(e) for e in eigh(A)[0])

def Fhat(J1, J2, dY, dX):
    return lmin(ptrace_Y(gmean(J1, J2), dY, dX))

def rand_c(n, m, rng):
    A = mp.zeros(n, m)
    for i in range(n):
        for j in range(m):
            A[i, j] = mp.mpc(rng.gauss(0, 1), rng.gauss(0, 1))
    return A

def rand_channel_kraus(dX, dY, k, rng):
    """random channel with k Kraus operators (Stiefel)."""
    G = rand_c(k*dY, dX, rng)
    # orthonormalise columns: V = G (G^*G)^{-1/2}
    V = G*fun(dag(G)*G, lambda t: 1/mp.sqrt(t))
    return [V[i*dY:(i+1)*dY, 0:dX] for i in range(k)]

def expm_herm(H, t):  # exp(-i t H)
    E, Q = eigh(H)
    n = H.rows; D = mp.zeros(n)
    for i in range(n): D[i, i] = mp.exp(-1j*t*E[i])
    return Q*D*dag(Q)

def rand_herm(n, rng):
    A = rand_c(n, n, rng); return herm(A)
