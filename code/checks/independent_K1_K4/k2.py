import mpmath as mp
from lib import *
mp.mp.dps=60
def closed(p,d):
    p=mp.mpf(p); d=mp.mpf(d)
    x=4*(1-p)**2*mp.sin(d/2)**2/(p*(4-3*p))
    return p/2+(1-p/2)/mp.sqrt(1+x)
worst=0
for p in ['0.1','0.01','0.001']:
    J0=mp_choi_Np(p)
    ev=mp_eigvals_h(J0)
    print('p',p,'eig J(N_p)',[mp.nstr(e,12) for e in ev], ' p/2=',mp.nstr(mp.mpf(p)/2,12))
    for d in ['1e-4','3e-4','1e-3','3e-3','1e-2','3e-2','0.1','0.3','0.6','1']:
        J1=mp_choi_Np(p,d)
        F=Fhat_mp(J0,J1); C=closed(p,d)
        rel=abs(F-C)/C; worst=max(worst,rel)
        print(f'  delta={d:6s} Fhat={mp.nstr(F,20)} closed={mp.nstr(C,20)} relerr={mp.nstr(rel,3)}')
print('worst relerr',mp.nstr(worst,3))
# delta* where F=0.9
for p in ['0.1','0.01','0.001','1e-4']:
    pm=mp.mpf(p)
    ds=mp.findroot(lambda d: closed(p,d)-mp.mpf('0.9'), 0.9687*mp.sqrt(pm))
    Fn=Fhat_mp(mp_choi_Np(p),mp_choi_Np(p,ds))
    print('p',p,'delta*',mp.nstr(ds,10),'delta*/sqrt(p)',mp.nstr(ds/mp.sqrt(pm),8),'Fhat numeric at delta*',mp.nstr(Fn,15))
# asymptotic constant: (1-p/2)/sqrt(1+x)=0.9-p/2 -> x = ((1-p/2)/(0.9-p/2))^2-1 ; x ~ delta^2/(4p)*... small p: 4 sin^2(d/2)/(4p)=d^2/(4p)
c=mp.sqrt(4*((1/mp.mpf('0.9'))**2-1)); print('p->0 limit delta*/sqrt(p) =',mp.nstr(c,10))
