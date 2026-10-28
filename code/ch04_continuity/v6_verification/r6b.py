import random, mlib as m, mpmath as mp
mp.mp.dps = 50
rng = random.Random(11)
# phi == 1 is exactly the equality case of the conjecture: LHS = ||C1 - C2|| = 2 gamma. Check Lemma S never exceeded
mx = 0
for _ in range(300):
    n = rng.choice([2,3,4]); gam = mp.mpf(rng.random())*mp.mpf('0.499')
    s = mp.sqrt(1-4*gam**2)
    mx = max(mx, (2*gam)/(2*mp.sqrt(2)*gam/mp.sqrt(1+s)))
print('phi=1: sup ratio', mp.nstr(mx, 30), '(must be < 1)')
