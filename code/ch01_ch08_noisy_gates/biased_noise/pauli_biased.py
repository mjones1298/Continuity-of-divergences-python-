"""Shared closed forms for biased Pauli noise (Theorem B specialised) and the Zhou-Jiang rate for Pauli channels.
Setting: N_delta = P_q o Ad_{R_P(delta)} (o Ad_U, irrelevant), R_P = exp(-i delta P/2).
For a single qubit with rates q = (q1, px, py, pz) and P in {X,Y,Z}: pair 1 = {1, P} (coupled rate q_P), pair 2 = spectator
pair (the other two Paulis).  Everything is expressed through the pairs (q, q'), T = q+q', h = 4qq'/T.
"""
import mpmath as mp

IDX = {'1': 0, 'X': 1, 'Y': 2, 'Z': 3}


def qubit_rates(pz, b, q_perp_split=(0.5, 0.5)):
    """Z-biased qubit: pz dominant, px + py = pz/b (bias b = pz/(px+py); b = 1/2 is depolarizing-like px=py=pz)."""
    pz = mp.mpf(pz); b = mp.mpf(b)
    s = pz / b
    px, py = s * q_perp_split[0], s * q_perp_split[1]
    return (1 - px - py - pz, px, py, pz)


def qubit_pairs(q, axis):
    q1, qx, qy, qz = q
    if axis == 'Z':
        return [(q1, qz), (qx, qy)]
    if axis == 'X':
        return [(q1, qx), (qy, qz)]
    if axis == 'Y':
        return [(q1, qy), (qx, qz)]
    raise ValueError(axis)


def xval(a, b, delta):
    s2 = mp.sin(mp.mpf(delta) / 2) ** 2
    return (a - b) ** 2 * s2 / (4 * a * b)


def g(x):
    return 1 - 1 / mp.sqrt(1 + x)


def phi(x):
    return mp.sqrt(x / (1 + x)) * mp.asinh(mp.sqrt(x))


def Fhat(pairs, delta):
    return mp.fsum((a + b) / mp.sqrt(1 + xval(a, b, delta)) for a, b in pairs)


def DBS(pairs, delta):
    return 2 * mp.fsum((a + b) * phi(xval(a, b, delta)) for a, b in pairs)


def Umegaki(pairs, delta):
    s2 = mp.sin(mp.mpf(delta) / 2) ** 2
    return s2 * mp.fsum((a - b) * mp.log(a / b) for a, b in pairs)


def Uhlmann(pairs, delta):
    s2 = mp.sin(mp.mpf(delta) / 2) ** 2
    return mp.fsum(mp.sqrt((a + b) ** 2 - (a - b) ** 2 * s2) for a, b in pairs)


# ---- Fisher informations (per use) ----
def F_RLD(pairs):
    return mp.fsum((a + b) * (a - b) ** 2 / (4 * a * b) for a, b in pairs)


def F_BKM(pairs):
    return mp.fsum((a - b) * mp.log(a / b) for a, b in pairs) / 2


def F_SLD(pairs):
    return mp.fsum((a - b) ** 2 / (a + b) for a, b in pairs)


def w_ZJ(pairs):
    return mp.fsum(4 * a * b / (a + b) for a, b in pairs if a * b > 0)


def F_inf(pairs):
    w = w_ZJ(pairs)
    return (1 - w) / w


def ratio(pairs):
    return F_RLD(pairs) / F_inf(pairs)


# ---- windows ----
def window_Fhat(pairs, eta):
    """delta_eta = inf{delta: 1 - Fhat >= eta}; 1 - Fhat is increasing on [0, pi] -> bisection."""
    eta = mp.mpf(eta)
    f = lambda d: 1 - Fhat(pairs, d) - eta
    lo, hi = mp.mpf(0), mp.pi
    if f(hi) < 0:
        return mp.inf
    for _ in range(mp.mp.prec + 10):
        mid = (lo + hi) / 2
        if f(mid) >= 0:
            hi = mid
        else:
            lo = mid
    return hi


def window_DBS(pairs, eta):
    eta = mp.mpf(eta)
    f = lambda d: DBS(pairs, d) - eta
    lo, hi = mp.mpf(0), mp.pi
    if f(hi) < 0:
        return mp.inf
    for _ in range(mp.mp.prec + 10):
        mid = (lo + hi) / 2
        if f(mid) >= 0:
            hi = mid
        else:
            lo = mid
    return hi


def ginv(y):
    y = mp.mpf(y)
    return (1 - y) ** -2 - 1


def window_bounds(pairs_main, T_rest, eta):
    """Two-sided bounds of Prop. (biased window): pair 1 = (q1, qP), T_rest = total weight of the other pairs.
    upper: T1 g(x1) >= eta;  lower (needs eta > T_rest): T1 g(x1) + T_rest >= eta."""
    q1, qP = pairs_main
    T1 = q1 + qP
    eta = mp.mpf(eta)

    def dl(xx):
        arg = 2 * mp.sqrt(q1 * qP * xx) / (q1 - qP)
        return 2 * mp.asin(arg) if arg <= 1 else mp.pi
    up = dl(ginv(eta / T1))
    lo = dl(ginv((eta - T_rest) / T1)) if eta > T_rest else mp.mpf(0)
    return lo, up
