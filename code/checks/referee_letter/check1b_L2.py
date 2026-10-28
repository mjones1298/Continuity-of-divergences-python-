# Referee check: Prop L2 p*Sigma_unit by golden-section search over log sin^2(theta)
import mpmath as mp
mp.mp.dps=40
for p in ['1e-2','1e-3','1e-4']:
    p=mp.mpf(p); beta,lam=2-3*p/2,p/2
    f=lambda s: mp.sqrt(2)/4*((1-s)/beta**2+s/lam**2)/((1-s)/beta+s/lam)**1.5
    g=lambda e: -f(mp.mpf(10)**e)
    best=min((g(mp.mpf(x)/400),mp.mpf(x)/400) for x in range(-3200,1))
    a,b=best[1]-mp.mpf(1)/200,best[1]+mp.mpf(1)/200
    gr=(mp.sqrt(5)-1)/2
    for _ in range(200):
        c=b-gr*(b-a); d=a+gr*(b-a)
        if g(c)<g(d): b=d
        else: a=c
    e=(a+b)/2; s=mp.mpf(10)**e
    print(f"p={mp.nstr(p,2)}  p*Sigma_unit={mp.nstr(-p*g(e),8)}  at sin^2(theta)={mp.nstr(s,6)} = {mp.nstr(s/lam,6)}*lam")
print("2/(3 sqrt3) =",mp.nstr(2/(3*mp.sqrt(3)),8))
