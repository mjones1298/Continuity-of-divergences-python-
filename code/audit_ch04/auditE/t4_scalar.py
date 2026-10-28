import numpy as np
rng=np.random.default_rng(0)
# scalar case: a1+b1+a2+b2=1, all >=0; T_i=a_i/(a_i+b_i)
N=4000000
x=rng.dirichlet([0.3,0.3,0.3,0.3],size=N)
a1,b1,a2,b2=x.T
with np.errstate(all='ignore'):
    T1=a1/(a1+b1);T2=a2/(a2+b2)
dl=np.maximum(abs(a1-a2),abs(b1-b2))
tau=abs(T1-T2)
ok=(dl<0.5)
rat=tau[ok]/(2*dl[ok]/(1-2*dl[ok]))
print('max ratio tau/(2d/(1-2d)) over random:',np.nanmax(rat))
# targeted: a1=a2+d*?, etc. grid search
best=0
for d in [0.4,0.3,0.2,0.1,0.05,0.01]:
    # a2=a1-e, b2=b1-f with |e|,|f|<=d ; C1=(1+e+f)/2
    bd=0
    g=np.linspace(-d,d,81)
    for e in g:
        for f in g:
            C1=(1+e+f)/2;C2=(1-e-f)/2
            if C1<=0 or C2<=0: continue
            for a1 in np.linspace(max(0,e),min(C1,C1),201):  # a2=a1-e>=0
                b1=C1-a1; a2=a1-e; b2=b1-f
                if a1<0 or b1<0 or a2<-1e-12 or b2<-1e-12: continue
                T1=a1/C1;T2=a2/C2
                bd=max(bd,abs(T1-T2))
    print(d,'max tau',bd,'bound 2d/(1-2d)',2*d/(1-2*d),'ratio',bd/(2*d/(1-2*d)))
