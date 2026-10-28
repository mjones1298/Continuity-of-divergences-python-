"""S3 (reduced S1): two-qubit CNOT with LOCAL depolarizing noise D_p (x) D_p after the gate.
Standard product-Pauli process tomography: 36 product input states x 9 product Pauli bases,
N shots per setting (4-outcome multinomial).  Estimator: linear inversion + CPTP projection.
Pairs: 'delta' = (est N_{p,theta}, est N_p) with U_theta = e^{-i theta Z(x)Z} CNOT, theta chosen so that
true F-hat = 0.71 (as for the qubit pair);  'self' = two independent estimates of N_p.
-> results/s3_cnot.json
"""
import os, sys, json, time, itertools, warnings
import numpy as np
import scipy.linalg as sla
import scipy.optimize as sopt
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
from pathlib import Path
sys.path.insert(0, str(Path(HERE).resolve().parents[1] / "common"))  # certlib.py, qlib.py in code/common
from certlib import INPUTS, BASES, _proj, herm, Q  # noqa

I2, X2, Y2, Z2 = Q.I2, Q.X2, Q.Y2, Q.Z2
ZZ = np.kron(Z2, Z2)


def choi2(p, theta):
    U = sla.expm(-1j * theta * ZZ) @ Q.CNOT
    kd = Q.kraus_depol(p)
    Ks = [np.kron(a, b) @ U for a in kd for b in kd]
    return Q.choi_from_kraus(Ks)


# design
SETS = list(itertools.product(range(6), range(6), range(3), range(3)))     # 324 settings
OUTS = list(itertools.product((+1, -1), (+1, -1)))                          # 4 outcomes
A = np.array([[np.kron(np.kron(_proj(BASES[m1], o1), _proj(BASES[m2], o2)),
                       np.kron(INPUTS[s1], INPUTS[s2]).T) for (o1, o2) in OUTS]
              for (s1, s2, m1, m2) in SETS])                                # (324,4,16,16)
P1 = [I2, X2, Y2, Z2]
HB = np.array([np.kron(np.kron(a, b), np.kron(c, d)) / 4.0 for a in P1 for b in P1 for c in P1 for d in P1])
AM = np.real(np.einsum("koab,jba->koj", A, HB)).reshape(-1, 256)
PINV = np.linalg.pinv(AM)


def probs(J):
    return np.clip(np.real(np.einsum("koab,ba->ko", A, J)), 0, None)


def sample(J, N, rng, reps):
    pr = probs(J)
    pr = pr / pr.sum(axis=1, keepdims=True)
    out = np.empty((reps,) + pr.shape)
    for k in range(pr.shape[0]):
        out[:, k, :] = rng.multinomial(int(N), pr[k], size=reps) / float(N)
    return out


def lin_inv(f):
    c = f.reshape(f.shape[0], -1) @ PINV.T
    return herm(np.einsum("rj,jab->rab", c, HB))


def ptrY(J):
    return np.einsum("...iaib->...ab", J.reshape(J.shape[:-2] + (4, 4, 4, 4)))


def tp_proj(J):
    S = ptrY(J) - np.eye(4)
    return J - 0.25 * np.einsum("ij,...ab->...iajb", np.eye(4), S).reshape(J.shape)


def psd_proj(J):
    w, U = np.linalg.eigh(herm(J))
    return herm(np.einsum("...ij,...j,...kj->...ik", U, np.clip(w, 0, None), U.conj()))


def tp_cong(J):
    w, U = np.linalg.eigh(herm(ptrY(J)))
    Sm = np.einsum("...ij,...j,...kj->...ik", U, 1 / np.sqrt(w), U.conj())
    K = np.einsum("ij,...ab->...iajb", np.eye(4), Sm).reshape(J.shape)
    return herm(K @ J @ K)


def cptp_project(J, iters=3000, tol=1e-12):
    X = J.copy(); P = np.zeros_like(J); Y = X
    for _ in range(iters):
        Y = psd_proj(X + P); P = X + P - Y; Xn = tp_proj(Y)
        if np.max(np.abs(Xn - X)) < tol:
            break
        X = Xn
    return tp_cong(psd_proj(Y))


def fhat(J1, J2):
    return Q.fhat_arn(J1, J2, 4, 4)


def dbs(J1, J2):
    return Q.dbs_arn(J1, J2, 4, 4)


PS = [1e-1, 3e-2, 1e-2]
NS = [10 ** e for e in np.arange(2, 11.01, 1.0)]
REPS = int(os.environ.get("REPS", 40))

if __name__ == "__main__":
    rng = np.random.default_rng(2026)
    out = dict(PS=PS, NS=NS, reps=REPS, rows=[])
    for p in PS:
        J0 = choi2(p, 0.0)
        th = sopt.brentq(lambda t: fhat(choi2(p, t), J0) - 0.7071, 1e-6, 1.0)
        J1 = choi2(p, th)
        tq = dict(p=p, theta=th, F=fhat(J1, J0), BS=dbs(J1, J0), lmin=float(np.linalg.eigvalsh(J0).min()))
        print(tq, flush=True)
        for N in NS:
            t0 = time.time()
            E = {k: cptp_project(lin_inv(sample(J, N, rng, REPS))) for k, J in (("d", J1), ("a", J0), ("b", J0))}
            Fd = [fhat(E["d"][i], E["a"][i]) for i in range(REPS)]
            Fs = [fhat(E["b"][i], E["a"][i]) for i in range(REPS)]
            Bd = [dbs(E["d"][i], E["a"][i]) for i in range(REPS)]
            lm = [float(np.linalg.eigvalsh(E["a"][i]).min()) for i in range(REPS)]
            out["rows"].append(dict(p=p, N=N, truth=tq, F_delta=Fd, F_self=Fs, BS_delta=Bd, lmin_a=lm))
            print(f"p={p} N={N:.0e} medF={np.median(Fd):.3f} medFself={np.median(Fs):.3f} "
                  f"BSinf={np.mean(np.isinf(Bd)):.2f} {time.time() - t0:.0f}s", flush=True)
            json.dump(out, open(os.path.join(HERE, "results", "s3_cnot.json"), "w"))
