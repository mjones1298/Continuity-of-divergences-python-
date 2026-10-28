import numpy as np
from scipy.integrate import quad
mu=lambda s: (2/np.pi)*np.sqrt(s)/(1+s)**2
def rhs(d):
    s0=d/(1-3*d) if d<1/3 else 1.0
    s0=min(s0,1.0)
    I1=quad(lambda s:(1/s)*mu(s),0,s0,limit=500)[0]+quad(lambda s:(1+3*s)*d/s**2*mu(s),s0,1,limit=500)[0]
    s1=1/d-3
    if s1>1:
        I2=quad(lambda s:(3+s)*d*mu(s),1,s1,limit=500)[0]+quad(lambda s:mu(s),s1,np.inf,limit=500)[0]
    else:
        I2=quad(lambda s:mu(s),1,np.inf,limit=500)[0]
    return I1+I2
for d in [1e-1,1e-2,1e-3,1e-4,1e-6,1e-8]:
    print(d,rhs(d),2*np.sqrt(d*(1-d)),rhs(d)/(2*np.sqrt(d*(1-d))))
