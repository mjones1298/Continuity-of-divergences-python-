# R6: sharper sandwich lemma. Independent test.
import random, mlib as m, mpmath as mp
mp.mp.dps = 50
rng = random.Random(7)

def rand_unitary(n):
    G = m.rand_c(n, n, rng); return G*m.fun(m.dag(G)*G, lambda t: 1/mp.sqrt(t))
def rand_effect(n):
    U = rand_unitary(n); D = m.zeros(n)
    for i in range(n): D[i, i] = mp.mpf(rng.random())
    return U*D*m.dag(U)

funcs = {
 'gsharp': (lambda t: mp.sqrt(max(t,0)*max(1-t,0)), lambda tau: mp.sqrt(tau), mp.mpf(1)/2),
 'id':     (lambda t: t, lambda tau: tau, mp.mpf(1)),
 'sq':     (lambda t: t*t, lambda tau: 2*tau, mp.mpf(1)),
 'const':  (lambda t: mp.mpf(1), lambda tau: 0*tau, mp.mpf(1)),
}
worst = {}; worst_second = 0; worst_v5 = 0; worst_conj = 0
for trial in range(1500):
    n = rng.choice([2, 3, 4])
    gam = mp.mpf(rng.random())*mp.mpf('0.499') * (rng.choice([1, 0.1, 0.01]))
    U = rand_unitary(n); D = m.zeros(n)
    lam = [mp.mpf(1)/2 + gam*(2*rng.random()-1) for _ in range(n)]
    lam[0] = mp.mpf(1)/2 + gam*rng.choice([1, -1])   # attain gamma
    for i in range(n): D[i, i] = lam[i]
    C1 = U*D*m.dag(U); C2 = m.eye(n) - C1
    a1 = m.fun(C1, mp.sqrt); a2 = m.fun(C2, mp.sqrt)
    same = rng.random() < 0.4
    T1 = rand_effect(n); T2 = T1 if same else rand_effect(n)
    tau = m.opnorm(T1 - T2)
    s = mp.sqrt(1 - 4*gam**2)
    for name, (f, om, nrm) in funcs.items():
        L = m.opnorm(a1*m.fun(T1, f)*a1 - a2*m.fun(T2, f)*a2)
        R = om(tau)/2 + 2*mp.sqrt(2)*gam*nrm/mp.sqrt(1+s)
        Rv5 = om(tau)/2 + 2*mp.sqrt(2)*gam*nrm
        Rc = om(tau)/2 + 2*gam*nrm
        worst[name] = max(worst.get(name, 0), L/R)
        worst_v5 = max(worst_v5, L/Rv5); worst_conj = max(worst_conj, L/Rc)
        if same: worst_second = max(worst_second, L/(2*mp.sqrt(2)*gam*nrm/mp.sqrt(1+s)))
print('max LHS/RHS (Lemma S) per phi:', {k: mp.nstr(v, 6) for k, v in worst.items()})
print('max LHS/RHS (v5 Lemma 10.3):', mp.nstr(worst_v5, 6))
print('max LHS/RHS (conjectured 2*gamma*||phi||):', mp.nstr(worst_conj, 6))
print('tau=0 instances, LHS/second term:', mp.nstr(worst_second, 6))

# Tightness of the second term: phi == 1, T1 = T2, gives LHS = ||C1-C2|| = 2 gamma; ratio = sqrt((1+s)/2) -> 1
for g in ['0.001', '0.1', '0.3', '0.45']:
    g = mp.mpf(g); s = mp.sqrt(1-4*g*g)
    print('gamma', g, 'phi=1 ratio', mp.nstr(2*g/(2*mp.sqrt(2)*g/mp.sqrt(1+s)), 8))

# Constant of improved Cor 10.7
def h_new(d):
    s = mp.sqrt(1-4*d*d)
    return mp.sqrt(2*mp.sqrt(2)/s) + 2*mp.sqrt(2)*mp.sqrt(d)/mp.sqrt(1+s)
def h_v5(d):
    s = mp.sqrt(1-4*d*d)
    return mp.sqrt(2*mp.sqrt(2)/s) + 2*mp.sqrt(2)*mp.sqrt(d)
for nm, h in [('new', h_new), ('v5-reopt', h_v5)]:
    d0 = mp.findroot(lambda d: h(d) - 1/mp.sqrt(d), mp.mpf('0.15'))
    print(nm, 'delta0 =', mp.nstr(d0, 12), ' K =', mp.nstr(1/mp.sqrt(d0), 12))
# monotonicity of h_new on (0,1/2)
xs = [mp.mpf(k)/1000 for k in range(1, 499)]
print('h_new increasing:', all(h_new(xs[i]) < h_new(xs[i+1]) for i in range(len(xs)-1)))
print('v5 constant at delta0=2/15 :', mp.nstr(h_v5(mp.mpf(2)/15), 8), mp.nstr(mp.sqrt(mp.mpf(15)/2), 8))
d0 = mp.mpf('0.1566')
print('new: h_new(0.1566)=', mp.nstr(h_new(d0), 8), ' 0.1566^{-1/2}=', mp.nstr(1/mp.sqrt(d0), 8))
