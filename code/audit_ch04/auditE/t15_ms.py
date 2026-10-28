import numpy as np
def kappa(s,t): return (2*t-1)**2/(s*(1-t)+t)
worst=0
for eps in [0.01,0.05,0.1,0.25,0.4,0.49]:
    rho=1-2*eps
    t=np.linspace(eps,1-eps,200001)
    for s in [0,1e-3,0.1,0.5,1,2,10,1e3]:
        k=kappa(s,t) if s>0 else (2*t-1)**2/t
        dk=np.gradient(k,t)
        M=np.abs(dk[1:-1]).max()
        ds=eps*(1+s)+rho*min(1,s)
        Ms=rho*(2/ds+(1+s)/ds**2)
        worst=max(worst,abs(M-Ms)/Ms)
        if abs(M-Ms)/Ms>1e-3: print('MISMATCH',eps,s,M,Ms)
print('worst rel',worst)
