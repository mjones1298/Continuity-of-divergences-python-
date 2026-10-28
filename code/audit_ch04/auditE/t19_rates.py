import numpy as np
from scipy.integrate import quad
# Hellinger: f=(sqrt x -1)^2, a=0,b=0,c=0, dmu = (2/pi) sqrt(s)/(1+s)^2 ds ; check total mass mu=?  f'(inf)=b+mu([0,inf)) should be 1
mu=lambda s: (2/np.pi)*np.sqrt(s)/(1+s)**2
print('mass',quad(mu,0,np.inf)[0],'int mu/s',quad(lambda s:mu(s)/s,0,1)[0])
# f(0+) = a-b+c+int 1/s dmu should be 1
print('f(0+)',quad(lambda s:mu(s)/s,0,np.inf)[0])
def rhs(d):
    I1=quad(lambda s:min((1+3*s)*d/s**2,1/s)*mu(s),0,1,limit=500)[0]
    I2=quad(lambda s:min((3+s)*d,1)*mu(s),1,np.inf,limit=500)[0]
    return I1+I2
for d in [1e-1,1e-2,1e-3,1e-4,1e-6]:
    print(d,rhs(d),2*np.sqrt(d*(1-d)),rhs(d)/ (2*np.sqrt(d*(1-d))))
