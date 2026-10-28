import numpy as np, warnings
from schur import schur_norm
from t_lem95 import schur2_formula
warnings.filterwarnings("ignore")
rng = np.random.default_rng(7)

def make(p, beta, w):
    wp = lambda t: 1/abs(t-p)
    g  = lambda t: beta*t + w*wp(t)
    zeta = 1.0 if p>0 else -1.0
    # derivative of w/|t-p|: = w*zeta/(t-p)^2 ... zeta=+1 if p>t_+
    gp = lambda t: beta + w*np.sign(p-t)*(1/(t-p)**2)
    return g, gp, wp

viol_a=0; viol_b=0; viol_c=0; cnt_c=0
for trial in range(80):
    tm = rng.uniform(-2,2); tp = tm + rng.uniform(0.05,3)
    side = rng.choice([-1,1])
    p = tp + rng.exponential(1.0)+0.01 if side>0 else tm - rng.exponential(1.0)-0.01
    beta = rng.normal()*rng.choice([0.1,1,10]); w = rng.normal()*rng.choice([0.1,1,10])
    g,gp,wp = make(p,beta,w)
    ends = np.array([tm,tp])
    M2 = np.array([[gp(tm), (g(tp)-g(tm))/(tp-tm)],[(g(tp)-g(tm))/(tp-tm), gp(tp)]])
    s2 = schur2_formula(M2[0,0],M2[0,1],M2[1,1])
    # sup over random Lambda and grids
    best=0
    for k in range(6):
        n = rng.integers(3,9)
        Lam = np.sort(rng.uniform(tm,tp,n)); 
        if k==0: Lam = np.linspace(tm,tp,9)
        if k==1: Lam = np.concatenate([[tm,tp], rng.uniform(tm,tp,5)])
        Mn = np.array([[ (g(x)-g(y))/(x-y) if abs(x-y)>1e-12 else gp(x) for y in Lam] for x in Lam])
        best = max(best, schur_norm(Mn))
    maxgp = max(abs(gp(tm)),abs(gp(tp)))
    if best > s2*(1+1e-6)+1e-8: viol_a+=1; print("(a) violation", trial, best, s2)
    if not (maxgp-1e-9 <= s2 <= np.sqrt(2)*maxgp+1e-9): viol_b+=1; print("(b) violation", trial, maxgp, s2)
    # (c) condition 1
    zeta = 1.0 if p>tp else -1.0
    cond1 = beta*w*(p-tm) >= 0
    # condition 2: root t0 of g'
    cond2=False
    ts = np.linspace(tm,tp,4001); gps = np.array([gp(t) for t in ts])
    idx = np.where(np.sign(gps[:-1])*np.sign(gps[1:])<=0)[0]
    for i in idx:
        from scipy.optimize import brentq
        t0 = brentq(gp, ts[i], ts[i+1]) if np.sign(gps[i])*np.sign(gps[i+1])<0 else ts[i]
        if wp(t0) <= 0.5*(wp(tm)+wp(tp))+1e-12: cond2=True
    if cond1 or cond2:
        cnt_c+=1
        if abs(s2-maxgp)>1e-7*max(1,maxgp): viol_c+=1; print("(c) violation", trial, s2, maxgp, cond1, cond2)
print("viol a,b,c:", viol_a, viol_b, viol_c, " (c) cases tested:", cnt_c)

# (d) sharpness
for z in [1e-1,1e-2,1e-3,1e-4,1e-6]:
    k  = lambda t: t + 2*z*z/(t+z)
    kp = lambda t: 1 - 2*z*z/(t+z)**2
    M2 = np.array([[kp(0),(k(1)-k(0))],[(k(1)-k(0)),kp(1)]])
    print(z, "max|k'|=",max(abs(kp(0)),abs(kp(1))), " Lip(ends)=", schur2_formula(M2[0,0],M2[0,1],M2[1,1]), "SDP grid:", schur_norm(np.array([[ (k(x)-k(y))/(x-y) if abs(x-y)>1e-12 else kp(x) for y in np.linspace(0,1,12)] for x in np.linspace(0,1,12)])))
